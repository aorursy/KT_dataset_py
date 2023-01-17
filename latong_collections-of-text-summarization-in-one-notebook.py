import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from  collections import OrderedDict


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
df=pd.read_csv("/kaggle/input/corowp/coroWP.csv")
df.head()
!pip install bert-extractive-summarizer
body=df['text_body'][0]
from summarizer import Summarizer
model = Summarizer()
result = model(body, min_length=50,max_length=100)
full0 = ''.join(result)

print(full0)
#GPT2
body=df['text_body'][0]
from summarizer import Summarizer,TransformerSummarizer
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(body, min_length=50, max_length=100))
print(full)
model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full2 = ''.join(model(body, min_length=60,max_length=100))
print(full2)
# load BART summarizer
import transformers
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
from transformers import pipeline
summarizer = pipeline(task="summarization")
summary = summarizer(body, min_length=60, max_length=100)
print (summary)
print(df['summary'][0])