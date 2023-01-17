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
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
train=train.drop('keyword',1)
train=train.drop('location',1)

test=test.drop('keyword',1)
test=test.drop('location',1)
y_train=train.iloc[:,-1]
x_train=train.iloc[:,:-1]
ps=PorterStemmer()
lemmatizer=WordNetLemmatizer()
def atcontain(text):
    ar=[]
    text=text.split()
    for t in text:
        if("@" in t):
            ar.append("TAGSOMEBODY")
        else:
            ar.append(t)
    return " ".join(ar)
def dataclean(data):
    corpus=[]
    for i in range(data.shape[0]):
        tweet=data.iloc[i,-1]
        tweet=atcontain(tweet)
        tweet=re.sub(r'http\S+', '', tweet)
        tweet=re.sub('[^a-zA-z]'," ",tweet)
        tweet=tweet.lower()
        tweet=word_tokenize(tweet)
#         tweet=[ps.stem(word) for word in tweet if word not in stopwords.words('english')]
        tweet=[lemmatizer.lemmatize(word) for word in tweet if word not in stopwords.words('english')]
        tweet=[word for word in tweet if word not in set(string.punctuation)]
        tweet=" ".join(tweet)
        corpus.append(tweet)
    return corpus
x_corpus_train=dataclean(x_train)
x_corpus_test=dataclean(test)
dic=defaultdict(int)
for text in x_corpus_train:
    text=text.split()
    for word in text:
        dic[word]=dic[word]+1
sorted_data=sorted(dic.items(), key=lambda x:x[1],reverse=True)
sorted_data[:20]
cv=TfidfVectorizer(max_features=8000)
x_train_vector=cv.fit_transform(x_corpus_train).toarray()
x_test_vector=cv.transform(x_corpus_test).toarray()
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train_vector,y_train)
print(model.score(x_train_vector,y_train))
y_pred=model.predict(x_test_vector)
y_pred
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=4,n_estimators=500,warm_start=True,max_depth=6,min_samples_leaf=2,max_features='auto',min_samples_split=3)
rfc.fit(x_train_vector,y_train)
print(rfc.score(x_train_vector,y_train))
y_pred=rfc.predict(x_test_vector)
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train_vector,y_train,early_stopping_rounds=5, 
             eval_set=[(x_train_vector,y_train)], 
             verbose=False)
print(xgb.score(x_train_vector,y_train))
y_pred=xgb.predict(x_test_vector)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train_vector,y_train)
print(reg.score(x_train_vector,y_train))
y_pred=reg.predict(x_test_vector)
y_pred
from sklearn.linear_model import PassiveAggressiveClassifier
passive=PassiveAggressiveClassifier()
passive.fit(x_train_vector,y_train)
print(passive.score(x_train_vector,y_train))
y_pred=passive.predict(x_test_vector)
y_pred
submission=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
submission['target']=y_pred
submission.to_csv('submission.csv',index=False)
