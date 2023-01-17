import numpy as np 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string as s
import re

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
cb_data= pd.read_csv('/kaggle/input/clickbait-dataset/clickbait_data.csv')
cb_data.head()
x=cb_data.headline
y=cb_data.clickbait
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=2)
print("No. of elements in training set")
print(train_x.size)
print("No. of elements in testing set")
print(test_x.size)
train_x.head(10)
train_y.head(10)
test_x.head()
test_y.head()
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
print("All stopwords of English language ")
", ".join(stopwords.words('english'))
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
train_x.head()
test_x.head()
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
print("Number of features extracted")
print(len(tfidf.get_feature_names()))
print()
print("The 100 features extracted from TF-IDF ")
print(tfidf.get_feature_names()[:100])

print("Shape of train set",train_1.shape)
print("Shape of test set",test_1.shape)
train_arr=train_1.toarray()
test_arr=test_1.toarray()
NB_MN=MultinomialNB()

NB_MN.fit(train_arr,train_y)
pred=NB_MN.predict(test_arr)
print('first 20 actual labels: ',test_y.tolist()[:20])
print('first 20 predicted labels: ',pred.tolist()[:20])
from sklearn.metrics import f1_score,accuracy_score
print("F1 score of the model")
print(f1_score(test_y,pred))
print("Accuracy of the model")
print(accuracy_score(test_y,pred))
print("Accuracy of the model in percentage")
print(accuracy_score(test_y,pred)*100,"%")
from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(test_y,pred))

from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(test_y,pred))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(ngram_range=(1,3),max_features=6500)
train_2=tfidf.fit_transform(train_x)
test_2=tfidf.transform(test_x)

NB_MN.fit(train_2.toarray(),train_y)
pred2=NB_MN.predict(test_2.toarray())

print("Accuracy of the model in percentage")
print(accuracy_score(test_y,pred2)*100,"%")

from sklearn.metrics import confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(test_y,pred2))

from sklearn.metrics import classification_report
print("Classification Report")
print(classification_report(test_y,pred2))