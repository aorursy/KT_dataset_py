# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib inline

import re

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize import TweetTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



import xgboost

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,SpatialDropout1D

from tensorflow.keras.layers import LSTM,Dropout

from keras.layers import Bidirectional

from tensorflow.keras.optimizers import RMSprop,Adam

from sklearn.model_selection import RandomizedSearchCV

from numpy import array

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.head()
test.head()
train["target"].value_counts()
sns.countplot("target",data=train)
from textblob import TextBlob
tweettoken = TweetTokenizer(strip_handles=True, reduce_len=True)
lemmatizer=WordNetLemmatizer()
stemmer=PorterStemmer()
collect=[]

collecttest=[]

def preprocess(t,kpc):

    def form_sentence(tweet):

        tweet_blob = TextBlob(tweet)

        return ' '.join(tweet_blob.words)

    t=form_sentence(t)

    tee=re.sub('[^a-zA-Z]'," ",t)

    tee=tee.lower()

    res=tweettoken.tokenize(tee)

    for i in res:

        if i in stopwords.words('english'):

            res.remove(i)

    rest=[]

    for k in res:

        rest.append(stemmer.stem(k))

    ret=" ".join(rest)

    if kpc==1:

        collect.append(ret)

    elif kpc==0:

        collecttest.append(ret)
def splitpro(t,q,m):

         for j in range(q):

                 preprocess(t["text"].iloc[j],m)
len(train)
len(test)
splitpro(train,7613,1)
splitpro(test,3263,0)
len(collect)
len(collecttest)
collect[:5]
collecttest[:5]
val=train["target"].values
val
def tfidf(xx):

    cv=TfidfVectorizer(max_features=10000)

    x=cv.fit_transform(xx).toarray()

    return x
y=tfidf(collect)
y[0]
len(y[0][:])
(x_train,x_test,y_train,y_test) = train_test_split(y,val, train_size=0.80, random_state=42)
rnd_clf=RandomForestClassifier(n_estimators=200,random_state=42)

rnd_clf.fit(x_train,y_train)

rnd_clf.score(x_test,y_test)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=0.1)

clf.fit(x_train,y_train)

clf.score(x_test,y_test)
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('rf', rnd_clf),('clf',clf)],voting='soft')

voting_clf.fit(x_train, y_train)
voting_clf.score(x_test,y_test)
ytt=tfidf(collecttest)
ttttt=voting_clf.predict(ytt)
r = pd.Series(ttttt,name="target")
sample=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample.head()
t=test["id"]
t
submiss = pd.concat([pd.Series(t,name = "id"),r],axis = 1)
submiss
submiss.to_csv("disasml20.csv",index=False)