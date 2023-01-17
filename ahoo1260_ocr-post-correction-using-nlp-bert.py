import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np 

import pandas as pd 

import os

import torch
os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM

from pytorch_pretrained_bert.modeling import BertModel
BERT_FP = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased/'
tokenizer = BertTokenizer(vocab_file='../input/torch-bert-weights/bert-base-uncased-vocab.txt')
from PIL import Image

from pytesseract import image_to_string

import re

import nltk

# from enchant.checker import SpellChecker

from difflib import SequenceMatcher

filename = '/kaggle/input/a-sample-for-ocr/sample.png'

text = image_to_string(Image.open(filename))

text_original = str(text)

print (text_original)
# lets tokenize some text (I intentionally mispelled 'plastic' to check berts subword information handling)

tokens = tokenizer.tokenize(text_original)

tokens
new_tokens=[]

i=0

while i<len(tokens):

    if tokens[i].startswith("##"):

        print(tokens[i])

        head=tokens[i-1]

        new_tokens=new_tokens[:-1]

        incorrect_str=head+tokens[i][2:]

        i=i+1

        while tokens[i].startswith("##"):

            incorrect_str=incorrect_str+tokens[i][2:]

            i=i+1

        

        new_tokens.append((incorrect_str,1))

    else:

        new_tokens.append((tokens[i],0))

        i=i+1

        

        

        
new_tokens
new_text_str=""

incorrect_words=[]

for word in new_tokens:

    if word[1]==0:

        new_text_str=new_text_str+word[0]+" "

    else:

        incorrect_words.append(word[0])

        new_text_str=new_text_str+'[MASK]'+" "
new_text_str
# Load, train and predict using pre-trained model

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_text = tokenizer.tokenize(new_text_str)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']

# Create the segments tensors

segs = [i for i, e in enumerate(tokenized_text) if e == "."]

segments_ids=[]

prev=-1

for k, s in enumerate(segs):

    segments_ids = segments_ids + [k] * (s-prev)

    prev=s

segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))

segments_tensors = torch.tensor([segments_ids])

# prepare Torch inputs 

tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model

model = BertForMaskedLM.from_pretrained(BERT_FP)



predictions = model(tokens_tensor, segments_tensors)

# Predict all tokens

with torch.no_grad():

    predictions = model(tokens_tensor, segments_tensors)
incorrect_words
text_original
torch.topk(predictions[0, MASKIDS[1]], k=50)
#Predict words for mask using BERT; 

#refine prediction by comparing with original word

import nltk



def predict_word(text_original, predictions, maskids):

    pred_words=[]

    for i in range(len(MASKIDS)):

        list2=[]

        preds = torch.topk(predictions[0, MASKIDS[i]], k=50) 

        indices = preds[1].tolist()

        list1 = tokenizer.convert_ids_to_tokens(indices)

        print(list1)

        for predicted in list1:

            dist=nltk.edit_distance(predicted,incorrect_words[i])

            list2.append((predicted,dist))

        

        sorted_list2 = sorted(list2, key=lambda tup: tup[1])

        final_predicted_word=sorted_list2[0]

        

        text_original=text_original.replace(incorrect_words[i],final_predicted_word[0])

    

    return text_original



            

        

predict_word(text_original, predictions, MASKIDS)
