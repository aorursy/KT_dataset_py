import os, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
train_data = pd.read_csv('/kaggle/input/zenify-tweet-train-folds/train_folds.csv')
train_data.head()
model_name = "roberta-base"
sanitycheck_model_names = ['bert-base-uncased','bert-large-uncased','bert-base-cased',
                    'bert-large-cased','bert-base-multilingual-uncased',
                    'bert-base-multilingual-cased','roberta-base','roberta-large',
                    'albert-base-v1','albert-large-v1','albert-xlarge-v1']
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

for train_index in range(5):
    question = train_data.sentiment[train_index]
    answer = train_data.text[train_index]
    encoded_input = tokenizer.encode_plus(question, answer, add_special_tokens=True, return_offsets_mapping=True)
    try:
        print("\nQuestion: " + question + ', Answer: ' + answer)
        print("Encoded Input: " + str(encoded_input['input_ids']))
        print("Attention Mask: " + str(encoded_input['attention_mask']))
        print("Offset: " + str(encoded_input['offset_mapping']))
        print("Token Type Ids: " + str(encoded_input['token_type_ids']))
    except:
        pass
    
for model_ in sanitycheck_model_names:
    print("\nModel: " + model_)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_, use_fast=True)
        question = train_data.sentiment[0]
        answer = train_data.text[0]
    
        if model_ in ["albert-base-v1", "albert-large-v1", "albert-xlarge-v1"]:
            encoded_input = tokenizer.encode_plus(question, answer, add_special_tokens=True, return_offsets_mapping=False)        
        else:
            encoded_input = tokenizer.encode_plus(question, answer, add_special_tokens=True, return_offsets_mapping=True)
            
        print("Question: " + question + ', Answer: ' + answer)
        print("Encoded Input: " + str(encoded_input['input_ids']))
        print("Attention Mask: " + str(encoded_input['attention_mask']))
        print("Offset: " + str(encoded_input['offset_mapping']))
        print("Token Type Ids: " + str(encoded_input['token_type_ids']))
    except:
        pass
model = AutoModel.from_pretrained(model_name)
for model_ in sanitycheck_model_names:
    print("\nModel: " + model_)
    try:
        model = AutoModel.from_pretrained(model_)
        print("Model successfully loaded!")
        del model
    except:
        pass