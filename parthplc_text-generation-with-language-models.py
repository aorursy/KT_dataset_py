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
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import ast

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('/kaggle/input/generatewitht5/result/')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)
def beam_search_decoding (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                         attention_mask=attn_mask,
                                         max_length=512,
                                       num_beams=20,
                                       num_return_sequences=1,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True
                                       )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
        return [Question.strip().capitalize() for Question in Questions]
def t5_answer(input_text):
    con = "Input: %s </s>" %(input_text)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = beam_search_decoding (input_ids, attention_masks)
    return output[0]
text = "Sachin is a great batsman and a"
t5_answer(text)
question = "Coronavirus is a deadly disease"
t5_answer(question)
def greedy_decoding (inp_ids,attn_mask):
    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
    return Question.strip().capitalize()
def t5_greedy(input_text):
    con = "Input: %s </s>" %(input_text)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = greedy_decoding (input_ids, attention_masks)
    return output
t5_answer(question)
def top_kp (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                         attention_mask=attn_mask,
                                         max_length=200,
                                      do_sample=True, 
                                     top_k=0
                                       )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
        return [Question.strip().capitalize() for Question in Questions]
def t5_topkp(input_text):
    con = "Input: %s </s>" %(input_text)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = top_kp (input_ids, attention_masks)
    return output[0]
question
t5_topkp(question)
def top_kp2 (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                         attention_mask=attn_mask,
                                        do_sample=True, 
    max_length=220, 
    top_k=0, 
    temperature=0.7
                                       )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
        return [Question.strip().capitalize() for Question in Questions]
def t5_topkp2(input_text):
    con = "Input: %s </s>" %(input_text)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = top_kp2 (input_ids, attention_masks)
    return output[0]
t5_topkp2(question)
def top_kp3 (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                         attention_mask=attn_mask,
                                        do_sample=True, 
    max_length=220, 
    top_k=50 
                                       )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
        return [Question.strip().capitalize() for Question in Questions]
def t5_topkp3(input_text):
    con = "Input: %s </s>" %(input_text)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = top_kp3 (input_ids, attention_masks)
    return output[0]
t5_topkp3(question)
def top_kp4 (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                         attention_mask=attn_mask,
                                        do_sample=True, 
    max_length=220, 
     top_p=0.92, 
    top_k=0
                                       )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
        return [Question.strip().capitalize() for Question in Questions]
def t5_topkp4(input_text):
    con = "Input: %s </s>" %(input_text)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = top_kp4 (input_ids, attention_masks)
    return output[0]
t5_topkp4(question)
def top_kp5 (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                         attention_mask=attn_mask,
                                        do_sample=True, 
    max_length=300, 
     top_p=0.92, 
    top_k=50,
                                     num_return_sequences=3
                                     
                                       )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
        return [Question.strip().capitalize() for Question in Questions]
def t5_topkp5(input_text):
    con = "Input: %s </s>" %(input_text)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = top_kp5 (input_ids, attention_masks)
    return output
t5_topkp5(question)
question = "Links In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
t5_topkp5(question)
text = "Apple is one of the largest"
t5_topkp5(text)
# import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
# encode context the generation is conditioned on
input_ids = tokenizer.encode('Coronavirus is a deadly disease', return_tensors='pt')

# generate text until the output length (which includes the context length) reaches 50
sample_outputs = model.generate(
    input_ids,
    do_sample=True, 
    max_length=220, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("Output:\n" + 100 * '-')
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
