import pandas as pd

import numpy as np

import matplotlib as plt
df =pd.read_csv('/kaggle/input/spam.csv',encoding= 'latin-1')
df.head()
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1, inplace= True)
df
data=df.rename(columns={'v1':'class','v2':'text'})
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
x= data['text']

y=data['class']

x_train,x_test, y_train,y_test= train_test_split(x,y,test_size=0.2)
v=CountVectorizer()

v.fit(x_train)

vec_x_train= v.transform(x_train).toarray()

vec_x_test= v.transform(x_test).toarray()
v.vocabulary
from sklearn.naive_bayes import MultinomialNB,GaussianNB, BernoulliNB
m= GaussianNB()

m.fit(vec_x_train,y_train)

print(m.score(vec_x_test,y_test))
sample = input('ask a question:')

vec = v.transform([sample]).toarray()

m.predict(vec)