Streamlit 网页界面 + 个人知识库问答系统
# 功能：上传 PDF，自动构建知识库，在线问答

import os
import shutil
import tempfile
import streamlit as st

# ====== 设置 HuggingFace 国内镜像 ======
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# 配置区 - 换成你自己的硅基流动 Key
# ============================================================
API_KEY = "sk-API"
BASE_URL = "https://api.siliconflow.cn/v1"
CHAT_MODEL = "deepseek-ai/DeepSeek-V3"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
CHROMA_PATH = "./chroma_db"
# ============================================================

# 页面设置
st.set_page_config(page_title="个人知识库问答系统", page_icon="📚")
st.title("📚 个人知识库问答系统")
st.caption("上传 PDF，构建你的专属知识库，开始提问吧！")


# ============================================================
# 初始化组件（缓存，只加载一次）
# ============================================================
@st.cache_resource
def load_embeddings():
    """加载本地嵌入模型"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource
def load_llm():
    """加载大模型"""
    return ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model_name=CHAT_MODEL,
        temperature=0.3
    )


embeddings = load_embeddings()
llm = load_llm()


# ============================================================
# 处理上传的 PDF，创建向量库
# ============================================================
def process_pdfs(uploaded_files):
    """将用户上传的 PDF 处理成向量数据库"""
    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp()
    all_docs = []

    for uploaded_file in uploaded_files:
        # 保存上传文件到临时目录
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 加载 PDF
        loader = PyPDFLoader(file_path)
        all_docs.extend(loader.load())

    # 清理临时文件夹
    shutil.rmtree(temp_dir)

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    split_docs = text_splitter.split_documents(all_docs)

    # 删除旧数据库，创建新数据库
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectorstore, len(split_docs)


# ============================================================
# 构建 RAG 链
# ============================================================
def build_rag_chain(vectorstore):
    """根据向量库构建问答链"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个基于文档知识的问答助手。
请严格根据以下提供的上下文来回答问题。
如果上下文没有足够的信息，请直接说"根据我目前的知识库，我无法回答这个问题"，不要编造任何内容。

上下文：
{context}"""),
        ("user", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return rag_chain


# ============================================================
# 侧边栏：文件上传
# ============================================================
with st.sidebar:
    st.header("📁 上传你的 PDF")
    uploaded_files = st.file_uploader(
        "选择 PDF 文件（可多选）",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("开始构建知识库"):
            with st.spinner("正在处理文档，请稍候..."):
                vectorstore, doc_count = process_pdfs(uploaded_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = build_rag_chain(vectorstore)
            st.success(f"知识库构建完成！共处理 {doc_count} 个文档块。")

    st.divider()
    st.caption("技术栈：LangChain + Chroma + Streamlit + DeepSeek")

# ============================================================
# 主界面：聊天窗口
# ============================================================
# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题"):
    # 检查是否已构建知识库
    if "rag_chain" not in st.session_state:
        st.warning("请先在侧边栏上传 PDF 并构建知识库！")
    else:
        # 显示用户消息
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 调用 RAG 链获取回答
        with st.spinner("思考中..."):
            try:
                answer = st.session_state.rag_chain.invoke(prompt)
            except Exception as e:
                answer = f"出错了：{e}"

        # 显示助手回答
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})