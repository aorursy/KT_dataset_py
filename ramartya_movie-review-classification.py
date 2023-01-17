import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

import string

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)





import os

print(os.listdir("../input"))

data=pd.read_csv("../input/dataset.csv")



data=data.iloc[:,1:]

# Any results you write to the current directory are saved as output.
data=data.sample(frac=1)
data.head()
train=data
train['clean_reviews']=train['reviews'].str.replace("[^a-zA-Z#]"," ")

train['clean_reviews'].head()
train['clean_reviews']=train['clean_reviews'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
train['clean_reviews'].head()
train['clean_reviews']=train['clean_reviews'].apply(lambda x: x.split())
train['clean_reviews'].head()
from nltk.stem.porter import *

stemmer=PorterStemmer()

train['clean_reviews']=train['clean_reviews'].apply(lambda x: [stemmer.stem(i) for i in x])
train['clean_reviews'].head()
train['clean_reviews']=train['clean_reviews'].apply(lambda x:' '.join([w for w in x]))

#train['clean_reviews']=join[x for x in train['clean_reviews']]
train['clean_reviews'].head(10)
from wordcloud import WordCloud
good_words=' '.join([text for text in train['clean_reviews'][train['label']==1]])

wc=WordCloud(height=500, width=500, random_state=21, max_font_size=110).generate(good_words)

plt.figure(figsize=(7,7))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
bad_words=' '.join([text for text in train['clean_reviews'][train['label']==0]])

wc=WordCloud(height=500, width=500, random_state=21, max_font_size=110).generate(bad_words)

plt.figure(figsize=(7,7))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
from sklearn.feature_extraction.text import HashingVectorizer

hv=HashingVectorizer(n_features=2**10)

hvr=hv.fit_transform(train['clean_reviews'])

fea_hvr=pd.DataFrame(hvr.toarray())
data_hvr=pd.concat([pd.DataFrame(fea_hvr),train['label']], axis=1)
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(min_df=2, max_df=0.90, max_features=1000,stop_words='english')# selects words that have min frequency above 2 and max frequnecy below 0.9*size of vocalubary.

#among these words, it chooses 1000 words with highest frequency

bow=cv.fit_transform(train['clean_reviews'])

#print(bow)
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['This is the first document.','This document is the second document.',

     'And this is the third one.',

     'Is this the first document?']

vec=TfidfVectorizer()

X=vec.fit_transform(corpus)

print(vec.get_feature_names())

print(X.shape)

print(X)



tv=TfidfVectorizer(min_df=2, max_df=0.90,  max_features=1000, stop_words='english')

tfidf=tv.fit_transform(train['clean_reviews'])

vocab=tv.get_feature_names()

print(len(vocab))# length of vocabulary

#print(tfidf)

fea_bow=pd.DataFrame(bow.toarray())

fea_tf=pd.DataFrame(tfidf.toarray())
from sklearn.decomposition import PCA, TruncatedSVD



x_tsvd_bow=TruncatedSVD(n_components=100,algorithm='randomized',random_state=42).fit_transform(fea_bow.values)

x_tsvd_tf=TruncatedSVD(n_components=100,algorithm='randomized',random_state=42).fit_transform(fea_tf.values)
data_bow=pd.concat([pd.DataFrame(x_tsvd_bow),train['label']], axis=1)

data_tf=pd.concat([pd.DataFrame(x_tsvd_tf),train['label']],axis=1)

data_bow.shape
corr_bow=data_bow.corr()

corr_tf=data_tf.corr()
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,24))

sns.heatmap(corr_bow, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)

ax1.set_title("BOW model", fontsize=14)



sns.heatmap(corr_tf, cmap='coolwarm_r', annot_kws={'size':23})

ax2.set_title('Tfidf Model', fontsize=14)

plt.show()


pos_corr=corr_bow.index[corr_bow['label'] >0.010].tolist()#these eatures have strong +ve correlation

neg_corr=corr_tf.index[corr_tf['label'] <-0.011].tolist()#these eatures have strong -ve correlation

print(pos_corr,neg_corr)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
tr_bow=bow



x_tr_bow, x_val_bow, y_tr_bow, y_val_bow=train_test_split(tr_bow, train['label'],random_state=42,test_size=0.3)

model_bow=LogisticRegression()

y_tr_bow=pd.DataFrame(y_tr_bow)

model_bow.fit(x_tr_bow,y_tr_bow)

ans=model_bow.predict(x_val_bow)

f1_bow=f1_score(ans, y_val_bow)

print(f1_bow)
train_tfidf = tfidf



x_tr_tf,x_val_tf,y_tr_tf,y_val_tf=train_test_split(train_tfidf,train['label'], test_size=0.3)

model_tf=LogisticRegression()

model_tf.fit(x_tr_tf,y_tr_tf)

y_pre_tf=model_tf.predict(x_val_tf)

f1_tf=f1_score(y_pre_tf,y_val_tf)

f1_tf

x_tr_tf_tsvd,x_val_tf_tsvd,y_tr_tf_tsvd,y_val_tf_tsvd=train_test_split(x_tsvd_tf,train['label'], test_size=0.3)

model_tsvd_tf=LogisticRegression()

model_tsvd_tf.fit(x_tr_tf_tsvd,y_tr_tf_tsvd)

y_pre_tf_tsvd=model_tsvd_tf.predict(x_val_tf_tsvd)

f1_tf_tsvd=f1_score(y_pre_tf_tsvd,y_val_tf_tsvd)

f1_tf_tsvd
train_tfidf = tfidf



x_tr_tf,x_val_tf,y_tr_tf,y_val_tf=train_test_split(train_tfidf,train['label'], test_size=0.3)

from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()

clf.fit(x_tr_tf,y_tr_tf)

y_pre_tf=clf.predict(x_val_tf)

f1_tf=f1_score(y_pre_tf,y_val_tf)

f1_tf
x_tr_hvr,x_val_hvr,y_tr_hvr,y_val_hvr=train_test_split(data_hvr, train['label'],test_size=0.3 )

model_hvr=LogisticRegression()

model_hvr.fit(x_tr_hvr,y_tr_hvr)

y_pr_hvr=model_hvr.predict(x_val_hvr)

f1_hvr=f1_score(y_pr_hvr,y_val_hvr)

f1_hvr