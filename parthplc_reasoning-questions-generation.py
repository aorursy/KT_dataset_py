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



model = T5ForConditionalGeneration.from_pretrained('/kaggle/input/something-exiting/result')

tokenizer = T5Tokenizer.from_pretrained('t5-base')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print ("device ",device)

model = model.to(device)
text = "Do i need to go for a legal divorce ? I wanted to marry a woman but she is not in the same religion , so i am not concern of the marriage inside church . I will do the marriage registered with the girl who i am going to get married . But legally will there be any complication , like if the other woman comes back one day , will the girl who i am going to get married now will be in trouble or Is there any complication ?"
context = "Context: %s </s>" % (text)



max_len = 256



encoding = tokenizer.encode_plus(context, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
def decoding (inp_ids,attn_mask):

    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)

    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return Question.strip().capitalize()
print (text)

# print ("\nGenerated Question: ",truefalse)



output = decoding(input_ids,attention_masks)

print ("\nQuestion: ",output)
text2 = "A woman had topped herself by jumping off the roof of the hospital she had just recently been admitted to. She was there because the first or perhaps latest suicide attempt was unsuccessful. She put her clothes on, folded the hospital gown and made the bed. She walked through the unit unimpeded and took the elevator to the top floor."

context = "Context: %s </s>" % (text2)



max_len = 256



encoding = tokenizer.encode_plus(context, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)



print (text2)

# print ("\nGenerated Question: ",truefalse)



output = decoding(input_ids,attention_masks)

print ("\nQuestion: ",output)
text2 = "Good Old War and person L : I saw both of these bands Wednesday night , and they both blew me away . seriously . Good Old War is acoustic and makes me smile . I really can not help but be happy when I listen to them ; I think it 's the fact that they seemed so happy themselves when they played ."

context = "Context: %s </s>" % (text2)



max_len = 256



encoding = tokenizer.encode_plus(context, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)



print (text2)

# print ("\nGenerated Question: ",truefalse)



output = decoding(input_ids,attention_masks)

print ("\nQuestion: ",output)