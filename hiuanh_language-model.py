import pandas as pd
import numpy as np

data = pd.read_csv("../input/pos-tagging/pos_1.csv")
data.head(10)
sentences=[]
i=0
while i<len(data):
    tmp=[]
    a=[]
    b=[]
    while (data.iloc[i,0])!="." and data.iloc[i,1] !='.':
        a.append(data.iloc[i,0])
        a.append(data.iloc[i,1])
        tmp.append(tuple(a))
        a=[]
        i=i+1
    tmp.append(tuple(('.', '.')))
    sentences.append(tmp)
    i=i+1

len(sentences)

from nltk.tag import DefaultTagger  
from nltk.tag import BigramTagger 
  
from nltk.corpus import treebank 
  
train_data =sentences[:32287]  
test_data = sentences[32287:]

  

import nltk
t0 = nltk.DefaultTagger('NN')  
t1 = nltk.UnigramTagger(train_data,backoff=t0)
t2 = nltk.BigramTagger(test_data,backoff=t1)  
print('training accuracy', t2.evaluate(train_data))
print('testing accuracy', t2.evaluate(test_data))
