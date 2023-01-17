# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc
#pd.options.display.max_colwidth=100
pd.set_option('display.max_colwidth',100)
# Any results you write to the current directory are saved as output.
gc.collect()
train=pd.read_csv('../input/train.tsv',sep='\t')
print(train.shape)
train.head()
test=pd.read_csv('../input/test.tsv',sep='\t')
print(test.shape)
test.head()
sub=pd.read_csv('../input/sampleSubmission.csv')
sub.head()
test['Sentiment']=-999
test.head()
df=pd.concat([train,test],ignore_index=True)
print(df.shape)
df.tail()
del train,test
gc.collect()
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation
import re
from bs4 import BeautifulSoup
def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus
df['clean_review']=clean_review(df.Phrase.values)
df.head()

df_train=df[df.Sentiment!=-999]
df_train.shape
df_test=df[df.Sentiment==-999]
df_test.drop('Sentiment',axis=1,inplace=True)
print(df_test.shape)
df_test.head()
del df
gc.collect()
# from sklearn.feature_extraction.text import CountVectorizer
# cv=CountVectorizer()
# c_train=cv.fit_transform(df_train.clean_review).toarray()
# print(c_train.shape)
# c_test=cv.transform(df_test.clean_review).toarray()
# print(c_test.shape)
# bow_df=pd.DataFrame(data=c_train,columns=cv.get_feature_names())
# bow_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(ngram_range=(1,2),max_df=0.95,min_df=10,sublinear_tf=True)
c2_train=tfidf.fit_transform(df_train.clean_review).toarray()
print(c2_train.shape)
c2_test=tfidf.transform(df_test.clean_review).toarray()
print(c2_test.shape)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
y=le.fit_transform(df_train.Sentiment.values)
#y=pd.get_dummies(y).values
y.shape
del df_train,df_test
gc.collect()
# from sklearn.model_selection import train_test_split
# X_train,X_val,y_train,y_val=train_test_split(c2_train,y,test_size=0.2)
# print(X_train.shape,y_train.shape)
# print(X_val.shape,y_val.shape)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(c2_train,y)
y_pred=lr.predict(c2_test)
sub.Sentiment=y_pred
sub.head()
sub.to_csv('submission.csv',index=False)