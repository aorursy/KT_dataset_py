!git clone https://github.com/minimaxir/gpt-2-simple

%cd gpt-2-simple
!pip install -r requirements.txt
import tensorflow as tf

print(tf.__version__)
import os

import requests

import gpt_2_simple as gpt2
model_name = "124M"

if not os.path.isdir(os.path.join("models", model_name)):

    print(f"Downloading {model_name} model...")

    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
file_name = "shakespeare.txt"

if not os.path.isfile(file_name):

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    data = requests.get(url)



    with open(file_name, 'w') as f:

        f.write(data.text)
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,

              file_name,

              model_name=model_name,

              steps=300)   # steps is max number of training steps
gpt2.generate(sess)