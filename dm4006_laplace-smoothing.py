import nltk

import pandas as pd

import numpy

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import RegexpTokenizer

import re

import gc

from  collections import Counter

import math

import random
file_0=open("../input/xab",mode='r',encoding='latin-1')

file_content=file_0.read()

file_0.close()

del(file_0)

gc.collect()
temp = re.sub(r'[^a-zA-Z0-9\-\s\.]*', r'', file_content)

temp = re.sub(r'(\-|\s+)', ' ', temp)

del(file_content)
sent_token=nltk.sent_tokenize(temp[0:4500])

del(temp)
random.shuffle(sent_token)

train=sent_token[0:28]

test=sent_token[28:32]
def create_token(input):

    res=[]

    for x1 in input:

        l=x1.split(" ")

        res.extend(l)

    return res    

def find_trigrams(input_list):

    trigram_list = []

    for i in range(len(input_list)-2):

        trigram_list.append((input_list[i], input_list[i+1],input_list[i+2]))

    return trigram_list



def compper(d,test,tokenl):

    per=1

    smoothing=1/tokenl

    #value=d.keys()

    length=len(test)

    for t in test:

        if t in d.keys():

            

            per=per*d[t]

            #print("y")

        else:

            per=per*smoothing

    #print(per,length)        

    perl=pow(1/per,1/length)

    #print(per)

    return perl,per

random.shuffle(train)

dev_set1=train[25:28]

train_set1=train[0:25]

dev_set2=train[2:5]

train_set2=train[0:2]+train[5:28]

dev_set3=train[19:21]

train_set3=train[0:19]+train[21:28]

dev_set4=train[16:19]

train_set4=train[0:16]+train[19:28]

dev_set5=train[13:16]

train_set5=train[0:13]+train[16:28]
train_set=[train_set1,train_set2,train_set3,train_set4,train_set5]

dev_set=[dev_set1,dev_set2,dev_set3,dev_set4,dev_set5]
restest=[]

resdev=[]

resdevl=[]

restestl=[]

test_set=create_token(test)

for i in range(5):

    train=create_token(train_set[i])

    dev=create_token(dev_set[i])

    dev1=find_trigrams(dev)

    word_count = Counter(train)

    bi=Counter(nltk.ngrams(train,2))

    tri=Counter(nltk.ngrams(train,3))

    utoken=set(train)

    tokenl=len(utoken)

    length=len(train)

    vocab=len(utoken)

    tint=dict()

    for x in utoken:

        for y in utoken:

            if x!=y:

                for z in utoken:

                    if x!=z and y!=z:

                        if tri.get((x,y,z))==None:

                            tc=0

                        else :

                            tc=tri.get((x,y,z))

                        if bi.get((x,y))==None:

                            bc=0

                        else :

                            bc=bi.get((x,y))





                        tint[x,y,z]=(tc+1)/(bc+vocab)



    p1,l1=compper(tint,dev1,tokenl)

    resdevl.append(l1)

    resdev.append(p1)

    p2,l2=compper(tint,test,tokenl)

    restest.append(p2)

    restestl.append(l2)
df=pd.DataFrame()

df["dev"]=resdev

df["test"]=restest
print("perplexity")

df
df1=pd.DataFrame()

df1["dev"]=resdevl

df1["test"]=restestl
print("likelihood")

df1