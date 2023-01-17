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
from sklearn.utils import shuffle

import re

import string

import nltk

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.metrics import plot_confusion_matrix,precision_score,recall_score

from sklearn.model_selection import cross_val_score,train_test_split

fake=pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")

true=pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
#making a news target column

fake['target']=1

true['target']=0
Combine=pd.concat([fake,true])

#shuffling the data 

df=shuffle(Combine)

df=df.reset_index()
print(f"shape of dataframe:{df.shape}")
df.isna().sum()
df.info()
df.head(10)
def clean_text(text):

    text=text.lower()

    text=re.sub(f"[{string.punctuation}]"," ",text)

    text=re.sub("\[.*?\]"," ",text)   

    text=re.sub("\w\d\w"," ",text)

    text=re.sub("\(.*?\)"," ",text)

    text=re.sub("[\"\"]"," ",text) 

    text=re.sub(' t ',' ',text)

    text=re.sub(' s ',' ',text)

    text=re.sub("[^a-zA-Z]"," ",text)

    return text
#applying the process

df["text"]=df['text'].apply(lambda x:clean_text(x))

df['title']=df['title'].apply(lambda x:clean_text(x))
df['title'][443:579]
X=df['title']

Y=df['target']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)
Count_vec=CountVectorizer(stop_words='english',max_features=60)
X_Count=Count_vec.fit_transform(x_train).toarray()
Count_vec.get_feature_names()[:12]
X_Count.shape
Count_df=pd.DataFrame(X_Count,columns=Count_vec.get_feature_names())
Count_df.head(10)
naive_bayes_Countvec=MultinomialNB()

scores_cout=cross_val_score(naive_bayes_Countvec,X_Count,y_train,cv=5,scoring='f1')
scores_cout
naive_bayes_Countvec.fit(X_Count,y_train)
naive_bayes_tfidf=MultinomialNB()

tfidf_vec=TfidfVectorizer(stop_words='english')

X_tf=tfidf_vec.fit_transform(x_train)

scores_tf=cross_val_score(naive_bayes_tfidf,X_tf,y_train,cv=5,scoring='f1')
scores_tf
naive_bayes_tfidf.fit(X_tf,y_train)
test_count=Count_vec.transform(x_test)

plot_confusion_matrix(naive_bayes_Countvec,test_count,y_test)

y_pred_count=naive_bayes_Countvec.predict(test_count)
print("Naive_Bayes_Model with Countvectorizer : \n")

print(f"precision:{precision_score(y_test,y_pred_count)} \n recall:{recall_score(y_test,y_pred_count)}")
test_tf=tfidf_vec.transform(x_test)

plot_confusion_matrix(naive_bayes_tfidf,test_tf,y_test)
y_pred_tf=naive_bayes_tfidf.predict(test_tf)

print("Naive_Bayes_model with Tfidf: \n")

print(f"precsion:{precision_score(y_test,y_pred_tf)} \n recall:{recall_score(y_test,y_pred_tf)}")