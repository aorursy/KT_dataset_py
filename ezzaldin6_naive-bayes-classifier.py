import numpy as np 

import pandas as pd

import nltk

from nltk.corpus import stopwords

from nltk.probability import FreqDist

import string as s

import re

from textblob import TextBlob

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/onion-or-not/OnionOrNot.csv')

df.head()
df.info()
x=df.text

y=df.label

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=1)
def tokenization(text):

    lst=text.split()

    return lst

train_x=train_x.apply(tokenization)

test_x=test_x.apply(tokenization)
def lowercasing(lst):

    new_lst=[]

    for i in lst:

        i=i.lower()

        new_lst.append(i)

    return new_lst

train_x=train_x.apply(lowercasing)

test_x=test_x.apply(lowercasing)    
def remove_punctuations(lst):

    new_lst=[]

    for i in lst:

        for j in s.punctuation:

            i=i.replace(j,'')

        new_lst.append(i)

    return new_lst

train_x=train_x.apply(remove_punctuations)

test_x=test_x.apply(remove_punctuations)      
def remove_numbers(lst):

    nodig_lst=[]

    new_lst=[]

    for i in lst:

        for j in s.digits:    

            i=i.replace(j,'')

        nodig_lst.append(i)

    for i in nodig_lst:

        if i!='':

            new_lst.append(i)

    return new_lst

train_x=train_x.apply(remove_numbers)

test_x=test_x.apply(remove_numbers)
def remove_stopwords(lst):

    stop=stopwords.words('english')

    new_lst=[]

    for i in lst:

        if i not in stop:

            new_lst.append(i)

    return new_lst

train_x=train_x.apply(remove_stopwords)

test_x=test_x.apply(remove_stopwords)  
def remove_spaces(lst):

    new_lst=[]

    for i in lst:

        i=i.strip()

        new_lst.append(i)

    return new_lst

train_x=train_x.apply(remove_spaces)

test_x=test_x.apply(remove_spaces)  
lemmatizer=nltk.stem.WordNetLemmatizer()

def lemmatzation(lst):

    new_lst=[]

    for i in lst:

        i=lemmatizer.lemmatize(i)

        new_lst.append(i)

    return new_lst

train_x=train_x.apply(lemmatzation)

test_x=test_x.apply(lemmatzation)
train_x=train_x.apply(lambda x: ''.join(i+' ' for i in x))

test_x=test_x.apply(lambda x: ''.join(i+' ' for i in x))
freq_dist={}

for i in train_x.head(20):

    x=i.split()

    for j in x:

        if j not in freq_dist.keys():

            freq_dist[j]=1

        else:

            freq_dist[j]+=1

freq_dist
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()

train_1=tfidf.fit_transform(train_x)

test_1=tfidf.transform(test_x)
train_arr=train_1.toarray()

test_arr=test_1.toarray()
NB_MN=MultinomialNB()

def select_model(x,y,model):

    scores=cross_val_score(model,x,y,cv=5,scoring='f1')

    acc=np.mean(scores)

    return acc

select_model(train_arr,train_y,NB_MN)
NB_MN.fit(train_arr,train_y)

pred=NB_MN.predict(test_arr)

print('first 15 actual labels: ',test_y.tolist()[:20])

print('first 15 predicted labels: ',[pred[:20]])
from sklearn.metrics import f1_score,accuracy_score

print(f1_score(test_y,pred))

print(accuracy_score(test_y,pred))