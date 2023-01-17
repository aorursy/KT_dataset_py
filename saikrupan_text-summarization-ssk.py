!pip --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org install flask-ngrok
!pip install git+https://www.github.com/huggingface/transformers.git
# # Remove directory if exists
# import os
# import shutil
# # Flask app directory
# app_dir = '/content/Bart_T5-summarization'
# if os.path.exists(app_dir):
#     shutil.rmtree(app_dir)
# Get the Flask app html
!git clone https://github.com/krupanss/Bart_T5-summarization.git
# Change directory to flask app
import os
os.chdir('/kaggle/working/Bart_T5-summarization')
!pwd
!python3 app.py -models 'BART,T5,PEGASUS-CNN'