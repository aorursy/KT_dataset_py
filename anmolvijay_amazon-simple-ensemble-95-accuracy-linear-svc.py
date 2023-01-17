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
import matplotlib as plt

import seaborn as sns

import spacy

from sklearn.model_selection import train_test_split

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import string

from string import punctuation

import collections

from collections import Counter

import xgboost as xgb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
data = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')

data.head()
data.columns
data.info()
data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime'],axis=1,inplace=True)

data.head()
data['text'] = data['reviewText'] + ' ' + data['summary']

data.drop(['reviewText', 'summary'],axis=1,inplace=True)

data.head()
def simple_rating(rating):

    if(int(rating ) <= 3):

        return 0

    else:

        return 1



    

data.overall = data.overall.apply(simple_rating)
data.sample(20)
data.overall.value_counts()
data.text = data.text.astype('str')
nlp = spacy.load("en_core_web_sm")

tokenizer = RegexpTokenizer(r'\w+')

lemmatizer = WordNetLemmatizer()

stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)



            

def lemm_fun(text):

    final_text = []

    for word in text.split():

        if word.lower() not in stop:

            lem = lemmatizer.lemmatize(word)

            final_text.append(lem.lower())

    return " ".join(final_text)





            

data.text = data.text.apply(lemm_fun)
data.head()
x = data['text']

y = data['overall']
from sklearn.feature_extraction.text import TfidfVectorizer

   

tf=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))

  

x=tf.fit_transform(x)  
x.shape
from imblearn.combine import SMOTETomek

from imblearn.under_sampling import NearMiss



smk = SMOTETomek(random_state=42 , sampling_strategy = 0.8)

x_new,y_new=smk.fit_sample(x,y)



y_new.value_counts()

x_train,x_test,y_train,y_test = train_test_split(x_new, y_new, test_size = 0.2 , random_state = 0)

#Naive Bayes

nb = MultinomialNB()



#model

nb_model = nb.fit(x_train, y_train)



#predict

nb_train_predict = nb.predict(x_train)

nb_test_predict = nb.predict(x_test)



#accuracy

nb_train_acc = accuracy_score(y_train, nb_train_predict)

nb_test_acc = accuracy_score(y_test,nb_test_predict)



print('nb train accuracy:', nb_train_acc)

print('nb test accuracy:', nb_test_acc)

#random forest

rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)



#model

rf_model = rf.fit(x_train, y_train)



#predict

rf_train_predict = rf.predict(x_train)

rf_test_predict = rf.predict(x_test)



#accuracy

rf_train_acc = accuracy_score(y_train, rf_train_predict)

rf_test_acc = accuracy_score(y_test,rf_test_predict)



print('rf train accuracy:', rf_train_acc)

print('rf test accuracy:', rf_test_acc)
#logistic regression

lr = LogisticRegression(random_state=0)



#model

lr_model = lr.fit(x_train, y_train)



#predict

lr_train_predict = lr.predict(x_train)

lr_test_predict = lr.predict(x_test)



#accuracy

lr_train_acc = accuracy_score(y_train, lr_train_predict)

lr_test_acc = accuracy_score(y_test,lr_test_predict)



print('logistic regression train accuracy:', lr_train_acc)

print('logistic regression test accuracy:', lr_test_acc)
#Linear SVC

lsvc =  LinearSVC()



#model

lsvc_model = lsvc.fit(x_train, y_train)



#predict

lsvc_train_predict = lsvc.predict(x_train)

lsvc_test_predict = lsvc.predict(x_test)



#accuracy

lsvc_train_acc = accuracy_score(y_train, lsvc_train_predict)

lsvc_test_acc = accuracy_score(y_test,lsvc_test_predict)



print('linear svc train accuracy:', lsvc_train_acc)

print('linear svc test accuracy:', lsvc_test_acc)
#XG Boost

xgbc = xgb.XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=8,

 min_child_weight=1,

 gamma=0.1,

 subsample=0.8,

 colsample_bytree=0.8,

 nthread=4,

 scale_pos_weight=1,

 seed= 0)



xgb_model=xgbc.fit(x_train, y_train)



#predict

xgb_train_predict = xgbc.predict(x_train)

xgb_test_predict = xgbc.predict(x_test)



#accuracy

xgb_train_acc = accuracy_score(y_train, xgb_train_predict)

xgb_test_acc = accuracy_score(y_test,xgb_test_predict)



print('xgb train accuracy:', xgb_train_acc)

print('xgb test accuracy:', xgb_test_acc)














