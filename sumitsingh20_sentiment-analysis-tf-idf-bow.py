import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

import nltk

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import word2vec

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
dd = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',encoding = "ISO-8859-1")

dd.columns = ['sentiment','id','date','query','special','text']

dd.head()
dd.drop(['id','date','query','special'],axis = 1,inplace = True)
df = dd.sample(100000)

df['Cleaned'] = df['text'].str.replace('@','')

df['Cleaned'] = df['Cleaned'].str.replace(r'http\S+','')

df['Cleaned'] = df['Cleaned'].str.replace('[^a-zA-Z]',' ')
stopwords = stopwords.words('english')
def remove_stopwords(text):

    clean_text=' '.join([word for word in text.split() if word not in stopwords])

    return clean_text
df['Cleaned'] = df['Cleaned'].apply(lambda text : remove_stopwords(text.lower()))

df['Cleaned'] = df['Cleaned'].apply(lambda x : x.split())
df.head()
sns.countplot(df.sentiment)
wordnet=WordNetLemmatizer()

df['Cleaned'] = df['Cleaned'].apply(lambda x : [wordnet.lemmatize(i) for i in x])
df['Cleaned'] = df['Cleaned'].apply(lambda x : ' '.join([w for w in x]))
df['Cleaned'] = df['Cleaned'].apply(lambda x : ' '.join([w for w in x.split()]))
df.head()
cv = CountVectorizer(max_features = 2500)

x = cv.fit_transform(df['Cleaned']).toarray()

x.shape
x_train,x_test,y_train,y_test = train_test_split(x,df['sentiment'],test_size = 0.2,random_state = 42)
%%time

model = RandomForestClassifier()

model.fit(x_train,y_train)
model.score(x_train,y_train)
model.score(x_test,y_test)
%%time

reg = LogisticRegression()

reg.fit(x_train,y_train)
reg.score(x_train,y_train)
reg.score(x_test,y_test)
tf = TfidfVectorizer(max_features = 2500)

z = tf.fit_transform(df['Cleaned']).toarray()

z.shape
z_train,z_test,y_train,y_test = train_test_split(z,df['sentiment'],test_size = 0.2,random_state = 42)
%%time

model1 = RandomForestClassifier()

model1.fit(z_train,y_train)
model1.score(z_train,y_train)
model1.score(z_test,y_test)
%%time

reg1 = LogisticRegression()

reg1.fit(z_train,y_train)
reg1.score(z_train,y_train)
reg1.score(z_test,y_test)
scores = pd.DataFrame({'Bow(RF)': model.score(x_test,y_test),

                       'Bow(LR)': reg.score(x_test,y_test),

                       'TF(RF)': model1.score(z_test,y_test),

                       'TF(LR)': reg1.score(z_test,y_test)},

                      index = [0])
scores
scores.T.plot(kind = 'bar')