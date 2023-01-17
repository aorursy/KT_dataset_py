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
text = "My name parth and want to become data scientist." ##Incorrect Text


text = "%s </s>" % (text)





max_len = 128



encoding = tokenizer.encode_plus(text, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)





def greedy_decoding (inp_ids,attn_mask):

    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=128)

    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return Question.strip().capitalize()



print ("Incorrect: ",text)

# print ("\nGenerated Question: ",truefalse)



output = greedy_decoding(input_ids,attention_masks)

print ("\nCorrect: ",output)
!pip install --upgrade fastpunct

# from fastpunct import FastPunct

# # The default language is 'en'

# fastpunct = FastPunct('en')

# fastpunct.punct(["oh i thought you were here", "in theory everyone knows what a comma is", "hey how are you doing", "my name is sheela i am in love with hrithik"], batch_size=32)

# # ['Oh! I thought you were here.', 'In theory, everyone knows what a comma is.', 'Hey! How are you doing?', 'My name is Sheela. I am in love with Hrithik.']
from fastpunct import FastPunct

# The default language is 'en'

fastpunct = FastPunct('en')

fastpunct.punct(["oh i thought you were here", "in theory everyone knows what a comma is", "hey how are you doing", "My name is sheela i am in love with hrithik"], batch_size=32)

# ['Oh! I thought you were here.', 'In theory, everyone knows what a comma is.', 'Hey! How are you doing?', 'My name is Sheela. I am in love with Hrithik.']
output =output.lower()
l = []

l.append(output)
l
fastpunct = FastPunct('en')

fastpunct.punct(l)