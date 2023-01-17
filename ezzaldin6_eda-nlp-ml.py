import numpy as np 

import pandas as pd 

import nltk

from nltk.corpus import stopwords

import string as s

import re

from textblob import TextBlob

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from xgboost import XGBClassifier

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

def data_info(d):

    print('number of variables: ',d.shape[1])

    print('number of tweets: ',d.shape[0])

    print('variables names: ')

    print(d.columns)

    print('variables data-types: ')

    print(d.dtypes)

    print('missing values: ')

    c=d.isnull().sum()

    print(c[c>0])
data_info(train_df)
data_info(test_df)
plt.figure(figsize=(8,8))

train_df['target'].value_counts().plot.pie(autopct='%.2f%%')

plt.title('Disaster or Not Distribution')

plt.ylabel('')

plt.show()
train_df[train_df['target']==1].loc[:4,'text']
train_disasters={'earthquake':0,'fire':0,'fires':0,'shelter':0}

test_disasters={'earthquake':0,'fire':0,'fires':0,'shelter':0}

for i,j in zip(train_df['text'],test_df['text']):

    if 'earthquake' in i.split():

        train_disasters['earthquake']+=1

    elif 'earthquake' in j.split():

        test_disasters['earthquake']+=1

    if 'fire' in i.split():

        train_disasters['fire']+=1

    elif 'fire' in j.split():

        test_disasters['fire']+=1

    if 'fires' in i.split():

        train_disasters['fires']+=1

    elif 'fires' in j.split():

        test_disasters['fires']+=1

    if 'shelter' in i.split():

        train_disasters['shelter']+=1

    elif 'shelter' in j.split():

        test_disasters['shelter']+=1
print('number of tweets that fires,fire and earthquacke mentioned in train data: ',train_disasters)

print('number of tweets that fires,fire and earthquacke mentioned in test data: ',test_disasters)
def tokenization(text):

    lst=text.split()

    return lst

train_df['text']=train_df['text'].apply(tokenization)

test_df['text']=test_df['text'].apply(tokenization)
def lowercasing(lst):

    new_lst=[]

    for i in lst:

        i=i.lower()

        new_lst.append(i)

    return new_lst

train_df['text']=train_df['text'].apply(lowercasing)

test_df['text']=test_df['text'].apply(lowercasing)    
def remove_punctuations(lst):

    new_lst=[]

    for i in lst:

        for j in s.punctuation:

            i=i.replace(j,'')

        new_lst.append(i)

    return new_lst

train_df['text']=train_df['text'].apply(remove_punctuations)

test_df['text']=test_df['text'].apply(remove_punctuations)            
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

train_df['text']=train_df['text'].apply(remove_numbers)

test_df['text']=test_df['text'].apply(remove_numbers)     
def remove_stopwords(lst):

    stop=stopwords.words('english')

    new_lst=[]

    for i in lst:

        if i not in stop:

            new_lst.append(i)

    return new_lst

train_df['text']=train_df['text'].apply(remove_stopwords)

test_df['text']=test_df['text'].apply(remove_stopwords)  
def remove_spaces(lst):

    new_lst=[]

    for i in lst:

        i=i.strip()

        new_lst.append(i)

    return new_lst

train_df['text']=train_df['text'].apply(remove_spaces)

test_df['text']=test_df['text'].apply(remove_spaces)  
'''

def correct_spelling(lst):

    new_lst=[]

    for i in lst:

        i=TextBlob(i).correct()

        new_lst.append(i)

    return new_lst

train_df['text']=train_df['text'].apply(correct_spelling)

test_df['text']=test_df['text'].apply(correct_spelling)  

'''
lemmatizer=nltk.stem.WordNetLemmatizer()

def lemmatzation(lst):

    new_lst=[]

    for i in lst:

        i=lemmatizer.lemmatize(i)

        new_lst.append(i)

    return new_lst

train_df['text']=train_df['text'].apply(lemmatzation)

test_df['text']=test_df['text'].apply(lemmatzation)  
train_df['text']=train_df['text'].apply(lambda x: ''.join(i+' ' for i in x))

test_df['text']=test_df['text'].apply(lambda x: ''.join(i+' ' for i in x))
train_df.head()
from sklearn.feature_extraction.text import CountVectorizer

vec=CountVectorizer(ngram_range=(1,2))

train_1=vec.fit_transform(train_df['text'])

test_1=vec.transform(test_df['text'])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(ngram_range=(1,2))

train_2=tfidf.fit_transform(train_df['text'])

test_2=tfidf.transform(test_df['text'])
def select_model(x,y):

    models=[{

        'name':'Multinomial Naive Bayes',

        'estimator':MultinomialNB(),

        'hyperparameters':{

            'alpha':np.arange(0.1,1,0.01)

        }}]

    for i in models:

        print(i['name'])

        gs=GridSearchCV(i['estimator'], param_grid=i['hyperparameters'], cv=10, scoring='f1')

        gs.fit(x,y)

        print(gs.best_score_)

        print(gs.best_params_)

select_model(train_2, train_df['target'])
clf_NB=MultinomialNB()

clf_NB.fit(train_2,train_df['target'])

pred=clf_NB.predict(test_2)

pred[:15]
submission_df = {"id":test_df['id'],

                 "target":pred}

submission = pd.DataFrame(submission_df)

submission.to_csv('submission_df.csv',index=False)