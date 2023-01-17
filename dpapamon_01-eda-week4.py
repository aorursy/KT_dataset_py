!curl -s https://course.fast.ai/setup/colab | bash
!pip uninstall torch torchvision -y
!pip install "torch==1.4" "torchvision==0.5.0"
output = "/kaggle/working"
input = "/kaggle/input"
!ls /kaggle/input
%reload_ext autoreload
%autoreload 2 
%matplotlib inline


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np 
import string
import pandas as pd
from fastai.vision import *
from fastai.imports import *
from fastai.metrics import error_rate
import torch
torch.cuda.is_available()
test_df = pd.read_csv(f'{input}/test.csv')
test_df.info()
test_df.head(10)
train_df = pd.read_csv(f'{input}/train_en.csv')
train_df.info()
train_df.head(10)
train_df['n_product_title_wds'] = train_df['product_title'].apply(lambda x: len(x.split()))

train_df['n_product_title_wds'].describe()
train_df['n_product_title_wds'].value_counts().plot(kind='hist')
train_df['n_product_title_len'] = train_df['product_title'].fillna('').apply(len)

train_df['n_product_title_len'].describe()
train_df['n_product_title_len'].value_counts().plot(kind='hist')
punct = set(string.punctuation)

emoji = set()
for s in test_df['product_title'].fillna('').astype(str):
    for c in s:
        if c.isdigit() or c.isalpha() or c.isalnum() or c.isspace() or c in punct:
            continue
        emoji.add(c)
print(''.join(emoji))
train_df['product_title_emoji'] = train_df['product_title'].fillna('').apply(lambda x: sum(c in emoji for c in x))
train_df['product_title_emoji'].describe()
train_cn_df = pd.read_csv(f'{input}/train_tcn.csv')
train_cn_df.info()
train_cn_df.head(10)
!pip install tensor2tensor
from transformers import MarianMTModel, MarianTokenizer
base_model = "xlm-mlm-xnli15-1024"
base_model = "t5-base"
model = AutoModelWithLMHead.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

inputs = tokenizer.encode("translate Chinese to English: 你好 世界", return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)

print(outputs)

!pip install sacremoses subword_nmt fairseq
!pip install fairseq
import torch
import fairseq

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'lightconv.glu.wmt17.zh-en', ... ]

# # Load a transformer trained on WMT'16 En-De
zh2en = torch.hub.load('pytorch/fairseq', 'lightconv.glu.wmt17.zh-en', tokenizer='moses', bpe='subword_nmt')

# # The underlying model is available under the *models* attribute
# assert isinstance(zh2en.models[0], fairseq.models.lightconv.LightConvModel)

# Translate a sentence

import torch
from transformers import XLMTokenizer, XLMWithLMHeadModel
tokenizer = XLMTokenizer.from_pretrained("xlm-mlm-tlm-xnli15-1024")
model = XLMWithLMHeadModel.from_pretrained("xlm-mlm-tlm-xnli15-1024")
inputs = tokenizer.encode("translate English to Chinese: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs.tolist()[0]))

zh2en.translate('你好 世界')
!pip install adaptnlp