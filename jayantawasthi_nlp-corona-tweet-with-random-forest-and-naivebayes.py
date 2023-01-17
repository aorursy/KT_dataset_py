# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import xgboost
from sklearn.model_selection import RandomizedSearchCV
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv",encoding='latin1')
train.head()
def drop(p):
    p.drop(["UserName","ScreenName","Location","TweetAt"],axis=1,inplace=True)
drop(train)
train.head()
train["Sentiment"].value_counts()
def rep(t):
        d={"Sentiment":{'Positive':0,'Negative':1,"Neutral":2,"Extremely Positive":3,"Extremely Negative":4}}
        t.replace(d,inplace=True)
rep(train)
train.head()
tweettoken = TweetTokenizer(strip_handles=True, reduce_len=True)
lemmatizer=WordNetLemmatizer()
stemmer=PorterStemmer()
collect=[]
def preprocess(t):
    tee=re.sub('[^a-zA-Z]'," ",t)
    tee=tee.lower()
    res=tweettoken.tokenize(tee)
    for i in res:
        if i in stopwords.words('english'):
            res.remove(i)
    rest=[]
    for k in res:
        rest.append(lemmatizer.lemmatize(k))
    ret=" ".join(rest)
    collect.append(ret)
    
for j in range(41157):
    preprocess(train["OriginalTweet"].iloc[j])
collect[:5]
def bow(ll):
    cv=CountVectorizer(max_features=200)
    x=cv.fit_transform(ll).toarray()
    return x
    
y=bow(collect)
y[:1]
len(y[0][:])
def tfidf(xx):
    cv=TfidfVectorizer(max_features=4000)
    x=cv.fit_transform(xx).toarray()
    return x
values=train["Sentiment"].values
values
(x_train,x_test,y_train,y_test) = train_test_split(y,values, train_size=0.75, random_state=42)
x_train
rnd_clf=RandomForestClassifier(n_estimators=200,random_state=42)

rnd_clf.fit(x_train,y_train)
rnd_clf.score(x_test,y_test)
y_pred=rnd_clf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
cm
a=[400,500,600,700,800,900,1000]
for i in a:
    rnd_clf=RandomForestClassifier(n_estimators=i,random_state=42)
    rnd_clf.fit(x_train,y_train)
    t=rnd_clf.score(x_test,y_test)
    print(t)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train,y_train)

clf.score(x_test,y_test)
y=tfidf(collect)
(x_train,x_test,y_train,y_test) = train_test_split(y,values, train_size=0.75, random_state=42)
rnd_clf=RandomForestClassifier(n_estimators=200,max_leaf_nodes=8,random_state=42)
rnd_clf.fit(x_train,y_train)
rnd_clf.score(x_test,y_test)
clf = MultinomialNB()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)