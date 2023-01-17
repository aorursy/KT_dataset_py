!pip install colab_host
import os
from colab_host import Host, SimpleHttpServer, JupyterNotebook, JupyterLab, FlaskApp, UvicornApp
# Simple Http Server:
os.chdir("/")
SimpleHttpServer(1000)
# Jupyter Notebook IDE
os.chdir("/")
JupyterNotebook(1000)
# Jupyter Lab IDE
os.chdir("/")
JupyterLab(1000)
# Flask or Gunicorn Application

os.chdir("/kaggle/")
FlaskApp(
    port=1000,
    app="main:app",
    git_url="https://github.com/PuneethaPai/colab_host_flask_demo.git",
    requirements_file="requirements.txt"
)
# FastAPI or Uvicorn Application

os.chdir("/kaggle/")

UvicornApp(
    port=1000,
    app="main:app",
    git_url="https://github.com/PuneethaPai/colab_host_uvicorn_demo.git",
    requirements_file="requirements.txt"
)
