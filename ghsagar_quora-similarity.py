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
train=pd.read_csv("/kaggle/input/quora-question-pairs/train.csv.zip")

print(train.shape)

test=pd.read_csv("/kaggle/input/quora-question-pairs/test.csv")

print(test.shape)

 
test.head()
# since there is 1 null value in question 1 and two null values in question 2 we change it to empty string 

train['question1']=train['question1'].fillna('')

train['question2']=train['question2'].fillna('')

y_train=train['is_duplicate']

train=train.drop(['is_duplicate'],axis=1)

test['question1']=test['question1'].fillna('')

test['question2']=test['question2'].fillna('')

 
train.head()
# adding related frequencies of the questions

 

train['freq_qid1'] =train.groupby('qid1')['qid1'].transform('count') 

train['freq_qid2']=train.groupby('qid2')['qid2'].transform('count')



# adding length of question 1 and question 2 

train['len_q1']=train['question1'].str.len()

train['len_q2']=train['question2'].str.len()







 
test['len_q1']=test['question1'].str.len()

test['len_q2']=test['question2'].str.len()

# adding number of words in each questions

train['n_words_q1']=train['question1'].apply(lambda x: len(x.split(" ")))

train['n_words_q2']=train['question2'].apply(lambda x: len(x.split(" ")))
test['n_words_q1']=test['question1'].apply(lambda x: len(x.split(" ")))

test['n_words_q2']=test['question2'].apply(lambda x: len(x.split(" ")))
# length of common words in two questions

def len_common_words(data):

    w1=set(map(lambda x: x.lower().strip(),data['question1'].split(" ")))

    w2=set(map(lambda x: x.lower().strip(),data['question2'].split(" ")))

    return 1.0* len(w1 & w2) 

train['len_c-woords']=train.apply(len_common_words,axis=1) 



# length of unique common words only

def combined_words(data):

    w1=set(map(lambda x: x.lower().strip(),data['question1'].split(" ")))

    w2=set(map(lambda x: x.lower().strip(),data['question2'].split(" ")))

    return 1.0* (len(w1)+ len(w2) )

train['combined_words']=train.apply(combined_words,axis=1) 



# words share is length of uniquewords divided by total combined words 

def words_share(data):

    w1=set(map(lambda x: x.lower().strip(),data['question1'].split(" ")))

    w2=set(map(lambda x: x.lower().strip(),data['question2'].split(" ")))

    return 1.0* (len(w1 & w2)/(len(w1)+ len(w2) ))

train['words_share']=train.apply(words_share,axis=1) 



# frequency of question1 + frequency of question

train['freq_q1q2']=train['freq_qid1'] + train['freq_qid2'] 



# absolute value of (question 1' frequency - question 2's frequency)



train['abs_diff']=abs(train['freq_qid1'] -train['freq_qid2'] )
test['len_c-woords']=test.apply(len_common_words,axis=1) 

test['combined_words']=test.apply(combined_words,axis=1) 

 
test.head()
test.head()

import re

from nltk.stem.porter import *

from bs4 import BeautifulSoup

import nltk



from nltk.corpus import stopwords

stopwords=stopwords.words("english")



# def text_preprocess(df):

#     cleaned=df['question1'].apply(lambda x :[ i for i in x if i not in stopwords])

#     return cleaned

import string

from bs4 import BeautifulSoup

train['question1'] = train['question1'].str.replace(r'[^\w\s]+', '')

 

train['question2']=train['question2'].str.replace(r'[^\w\s]+', '')



train['question1'] = [BeautifulSoup(text).get_text() for text in train['question1'] ]

train['question2'] = [BeautifulSoup(text).get_text() for text in train['question2'] ]
test['question1'] = test['question1'].str.replace(r'[^\w\s]+', '')

 

test['question2']=test['question2'].str.replace(r'[^\w\s]+', '')



test['question1'] = [BeautifulSoup(text).get_text() for text in test['question1'] ]

test['question2'] = [BeautifulSoup(text).get_text() for text in test['question2'] ]
from nltk.stem import PorterStemmer

def stem_sentences(sentence):

    porter_stemmer = PorterStemmer()

    tokens = sentence.split()

    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]

    return ' '.join(stemmed_tokens)



train['question1'] = train['question1'].apply(stem_sentences)

train['question2'] = train['question2'].apply(stem_sentences)

test['question1'] = test['question1'].apply(stem_sentences)

test['question2'] = test['question2'].apply(stem_sentences)
#Now performing advanced featurizations 

def featurization(q1,q2):

    features=[0,0,0,0,0,0,0,0]

    q1=q1.split()

    q2=q2.split()

    

    if len(q1) == 0 or len(q2) == 0:

        return features

    

    #without stop words

    q1_nostop=set([each for each in q1 if each not in stopwords])

    q2_nostop=set([each for each in q2 if each not in stopwords])

    

    #with stop words

    q1_stop=set([each for each in q1 if each in stopwords])

    q2_stop=set([each for each in q2 if each in stopwords])

    #common non-stop words

    common_non_stop_c=len(q1_nostop.intersection(q2_nostop))

    

    #length of common stop words

    common_stop_c=len(q1_stop.intersection(q2_stop))

    

    # len of common unique words count from all together

    common_all_c=len(set(q1).intersection(set(q2)))

    

    # cmin,cmax for without stop words

    cmin_nostop=common_non_stop_c/(min(len(q1_nostop),len(q2_nostop))+0.0000001)

    cmax_nostop=common_non_stop_c/(max(len(q1_nostop),len(q2_nostop))+0.0000001)

    

    #cmin,cmax with stop words

    cmin_stop=common_stop_c/(min(len(q1_stop),len(q2_stop)) +0.0000001)

    cmax_stop=common_stop_c/(max(len(q1_stop),len(q2_stop))+0.0000001)

    

    #cminall,cmaxall

    cmin_all=common_all_c/(min(len(q1),len(q2))+0.0000001)

    cmax_all=common_all_c/(max(len(q1),len(q2))+0.0000001)

    

    last_same=int(q1[-1]==q2[-1])

    first_same=int(q1[0]==q2[0])

    

    

    features[0]=cmin_nostop

    features[1]=cmax_nostop

    features[2]=cmin_stop

    features[3]=cmax_stop

    features[4]=cmin_all

    features[5]=cmax_all

    features[6]=first_same

    features[7]=last_same

    return features

    

    

    
from fuzzywuzzy import fuzz


def get_features(dataframe):

    

    token_features = dataframe.apply(lambda x: featurization(x["question1"], x["question2"]), axis=1)

    

    dataframe["cmin_nostop"]    = list(map(lambda x: x[0], token_features))

    dataframe["cmax_nostop"]   = list(map(lambda x: x[1], token_features))

    dataframe["cmin_stop"]     = list(map(lambda x: x[2], token_features))

    dataframe["cmax_stop"]      = list(map(lambda x: x[3], token_features))

    dataframe["cmin_all"]      = list(map(lambda x: x[4], token_features))

    dataframe["cmax_all"] = list(map(lambda x: x[5], token_features))

    dataframe["first_same"]  = list(map(lambda x: x[6], token_features))

    dataframe["last_same"] = list(map(lambda x: x[7], token_features))

    

    dataframe["fuzz_ratio"]            = dataframe.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)

    dataframe["token_set_ratio"]       = dataframe.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)

    dataframe["token_sort_ratio"]      = dataframe.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)

    

    

    return dataframe





train=get_features(train)



 

    


test=get_features(test)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

 

# questions = list(train['question1']) + list(train['question2'])

 

 

 



# tfidf = TfidfVectorizer(min_df=1)

# tfidf.fit_transform(questions)

#  print(tfidf.get_feature_names())
# q1vec=tfidf.transform(train['question1'])

# q2vec=tfidf.transform(train['question2'])
# from scipy.sparse import hstack

# from scipy.sparse import coo_matrix, hstack

# import scipy.sparse as sp

# combined=hstack((q1vec,q2vec))

# print(type(combined))

# print(type(train))

# # a=combined.todense()

# # print(type(a))

if 'question1' or 'question2' in (train.coulmns or test.columns):

    train=train.drop(['question1','question2'],axis=1)

    test=test.drop(['question1','question2'],axis=1)
# final_data=hstack((train, combined),format="csr",dtype='float64')
# len(tfidf.get_feature_names())

# tfidf = TfidfVectorizer()

# tfidf.fit_transform(list(train['question1'][10])+list(train['question2'][1]))

# print(tfidf.get_feature_names())

# total=list(train['question1'][0])+list(train['question2'][0])

# total

# train['question2'][0]

# t=['what is the step by step guid to invest in share market in india','what is the step by step guid to invest in share market']

# q1=['what is the step by step guid to invest in share market']

# q2=['what is the step by step guid to invest in share market']

# tfidf = TfidfVectorizer()

# tfidf.fit_transform(t)

# print(tfidf.get_feature_names())

# result=tfidf.transform(q1)

# print(result.toarray())

# print("*"*16)

# re=tfidf.transform(q2)

# print(re.toarray())

# train.head()
# from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# model = KNeighborsClassifier(n_neighbors=5)

# model.fit(final_data,y_train)

# predicted=model.predict(final_data)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(train,y_train)

predicted=model.predict(train)

score=accuracy_score(predicted,y_train)

print("the accuracy socre is : ",score)

score=accuracy_score(predicted,y_train)

print("the accuracy socre is : ",score)
test.columns
train=train.drop(['qid1','qid2','freq_qid1','freq_qid2','freq_q1q2','words_share'],axis=1)

 
train.columns
train=train.drop(['abs_diff'],axis=1)
fpredict=model.predict(test)
# from sklearn.metrics import log_loss

# logloss=log_loss(predicted,y_train)

# print("the  log loss is : ", logloss)

result=pd.DataFrame({'test_id':test.test_id,'is_duplicate':fpredict})

result.to_csv('mysubmission.csv', index=False)  

print("success!!!")
