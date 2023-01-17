import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Trained model is loaded here...



import torch

from transformers import T5ForConditionalGeneration,T5Tokenizer

import ast



def set_seed(seed):

  torch.manual_seed(seed)

  if torch.cuda.is_available():

    torch.cuda.manual_seed_all(seed)



set_seed(42)



model = T5ForConditionalGeneration.from_pretrained('../input/sentisum-eda-training/result')

tokenizer = T5Tokenizer.from_pretrained('t5-base')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print ("device ",device)

model = model.to(device)
# Topkp decoding function for generating the output



def topkp_decoding (inp_ids,attn_mask):

      topkp_output = model.generate(input_ids=inp_ids,

                                     attention_mask=attn_mask,

                                     max_length=50,

                                   do_sample=True,

                                   top_k=40,

                                   top_p=0.80,

                                   num_return_sequences=3,

                                    no_repeat_ngram_size=2,

                                    early_stopping=True

                                   )

      Questions = [tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]

      return list(Question.strip().capitalize() for Question in Questions)    
def t5_answer(review):

    set_seed(42)

    con = "Comment: %s </s>" %(review)

    encoding = tokenizer.encode_plus(con, return_tensors="pt")

    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    output = topkp_decoding(input_ids, attention_masks)

    output.sort(key=len, reverse=True)



    return output
# Negative sentiments must be given more importance since they are the deciding factor for any organization. In this function, all negative sentiments (if any) are pulled in front and best sentiments are refined.



def refined_out(out):

    temp1= list(out[0].split(', '))

    

    if len(out)==3:

        temp2= list(out[1].split(', ')) + list(out[2].split(', '))

        for el in temp2:

            if el.lower() not in map(str.lower, temp1) and len(temp1)<4:

                temp1.append(el)

    i=0

    j=len(temp1)-1

    

    while(j>i):

        if 'negative' in temp1[i] and 'negative' in temp1[j]:   

            i+=1



        elif 'positive' in temp1[i] and 'negative' in temp1[j]:

            temp1[i], temp1[j]= temp1[j], temp1[i]

            i+=1

            j-=1



        elif 'positive' in temp1[i] and 'positive' in temp1[j]:

            j-=1

        else:

            j-=1

    

    return ' '.join(temp1)
# Smoothed Bleu Score



import nltk

from nltk.translate.bleu_score import sentence_bleu

from nltk.translate.bleu_score import SmoothingFunction



def bleu_score(ref, pred):

    smoother = SmoothingFunction()

    score= sentence_bleu(ref,pred, smoothing_function=smoother.method4, weights= (0.8,0.2))

    

    return score
# Jaccard Similarity



def jaccard_similarity(list1, list2):

    s1 = set(list1)

    s2 = set(list2)

    return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
# A custom scorer that consider the encoding ID of each word and uses jaccard metrics for calculation 



def custom_scorer(val, out):

    encoding1 = tokenizer.encode_plus(val, return_tensors="pt")

    encoding2 = tokenizer.encode_plus(out, return_tensors="pt")

    ids1, attention_masks = encoding1["input_ids"].to(device), encoding1["attention_mask"].to(device)

    ids2, attention_masks = encoding2["input_ids"].to(device), encoding2["attention_mask"].to(device)

    ids1 = [ids1.tolist()[0]]

    ids2 = ids2.tolist()[0]

    

    score= bleu_score(ids1, ids2)

    

    return score
# Universal Sentence Encoder for measuring Cosine similarity between validation and test output. USE can maintain the semantic relation in the sentences.



import tensorflow as tf

import tensorflow_hub as hub

import numpy as np

import os, sys

from sklearn.metrics.pairwise import cosine_similarity

from numpy import dot

from numpy.linalg import norm



module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

embed = hub.load(module_url)
def cosine_similarity(val, out):



    val= embed(val).numpy()

    out= embed(out).numpy()

    cos_sim = dot(val[0],out[0])/(norm(val[0])*norm(out[0]))

    

    return cos_sim
ref= pd.read_csv('../input/sentisum-eda-training/ref.csv')
ref.head(3)
# Evaluation loop



import statistics



score=[]

for el in ref.index:

    out= t5_answer(ref['Review0'][el])

    out1= [refined_out(out)]

    #out1= list(out1.split(' '))

    val1= list(ref['Column16'][el].split(', '))

    val1= [' '.join(val1)]

    #val1= [list(val1.split(' '))]

    

    sc= cosine_similarity(val1, out1)

    score.append(sc)

  

avgscore= statistics.mean(score)
# Just a test...



out= t5_answer(ref['Review0'][16])
ref['Review0'][16]
out= refined_out(out)

out
print('The average Cosine similarity score of unseen validation dataset is: {}%'.format(avgscore*100))