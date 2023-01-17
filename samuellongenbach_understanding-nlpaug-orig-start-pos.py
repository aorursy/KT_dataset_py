!pip install --upgrade pip

!pip install ../input/nlpaug-0-0-14/nlpaug-master

!pip uninstall librosa --y
import numpy as np

import pandas as pd



import torch

import transformers



import nlpaug.augmenter.word as naw
TOPK=20 

ACT = 'substitute' #'insert'

aug_bert = naw.ContextualWordEmbsAug(

    model_path='bert-base-uncased',

    action=ACT, top_k=TOPK,include_detail=True,aug_p=0.6)
original_text = "testttt 123 http://curious.org"



output = aug_bert.augment(original_text)

new_text = output[0]

swaps = output[1]



print("Original:", original_text)

print("Length:", len(original_text))

print(" ")

print("New:",new_text)

print("Length:", len(new_text))

print(" ")



for s in swaps:

    print(s)
a = original_text

b = new_text

for i in range(max(len(a),len(b))):

    try:

        print(i,a[i],b[i])

    except:

        if len(b) > len(a):

            print(i,"~",b[i])

        else:

            print(i,a[i],"~")