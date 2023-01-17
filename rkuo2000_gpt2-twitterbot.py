!git clone https://github.com/minimaxir/gpt-2-simple
%cd gpt-2-simple
!pip install -r requirements.txt
import gpt_2_simple as gpt2
import os
import requests
model_name = "355M" # "124M", "355M", "774M", "1.5B"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
textfile = '../../input/tweets/DonaldTrump.txt'

with open(textfile,'r') as f1:
    text = f1.read()
import re
text=re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", text)
print(text)
with open('realDonaldTrump.txt','a') as f2:
    f2.write(text)
file_name = 'realDonaldTrump.txt'
sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=100)   # steps is max number of training steps
!ls checkpoint/run1
gpt2.generate(sess,
              length=10,
              temperature=0.7,
              prefix='<|startoftext|>',
              truncate='<|endoftext|>',
              include_prefix=False,
              nsamples=20,
              batch_size=20
              )
from datetime import datetime
gen_file = 'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())

gpt2.generate_to_file(sess,
                      destination_path=gen_file,
                      length=10,
                      temperature=1.0,
                      top_p=0.9,
                      prefix='<|startoftext|>',
                      truncate='<|endoftext|>',
                      include_prefix=False,
                      nsamples=20,
                      batch_size=20
                      )
