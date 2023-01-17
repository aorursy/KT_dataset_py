# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
qanda = pd.read_excel("/kaggle/input/validation/qanda.xls")
text = 'My is name Parth and want to I become Data a Scientist '
import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('/kaggle/input/t5-text-to-text-transformer/result/')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)
jumble = "Jumbled: %s </s>" % (text)
import gc
def decoder(unresolvedanswer,model,device,tokenizer):
#         self.set_seed(42)
        
        print ("device ",device)
        model = model.to(device)
        print('Line1')
#         tokenizer = T5Tokenizer.from_pretrained('t5-base')
        print('Line2')
        text = "%s </s>" % (unresolvedanswer)        
        print(unresolvedanswer)
        encoding = tokenizer.encode_plus(text,padding= True,truncation = True,return_tensors="pt")
        print('Line3')
        inp_ids, attn_mask = encoding["input_ids"], encoding["attention_mask"]
        print('Line4')
        greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=128)
        print('Line5')
        finalsentence =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)      
        print('Line6')
        gc.collect()
        return finalsentence.strip().capitalize()
Final = str(decoder(jumble,model,device,tokenizer))
print(Final)
!pip install --upgrade fastpunct

from fastpunct import FastPunct
l=[Final]
fastpunct = FastPunct('en')
fastpunct.punct(l)
