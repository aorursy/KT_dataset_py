#This submission is based on NLP workshop by the PyData team from Tel Aviv! Thanks a lot!

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys, re, collections, string, itertools

from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from gensim.models import Word2Vec

print(os.listdir("../input"))
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')

train_data.head() #showing some sample toxic comments; 
#learning that every toxic comment is tagged as 'toxic' with our without some sub-tag

X=train_data['comment_text']
X_test=test_data['comment_text']

labels=train_data.columns.values[2:]
toxic_sublabels=train_data.columns.values[3:]

ys=train_data[labels]
y0=train_data['toxic']
toxic_ys=train_data[toxic_sublabels][y0==1]
toxic_comments=X[y0==1]

toxic_comments.head()
toxic_ys.head()
test_data.head()
def clean_text(text):
    text=text.str.lower()
    digits = re.compile(r"\d[\d\.\$]*")
    not_allowed = re.compile(r"[^\s\w<>_]")
    text=text.str.replace(digits,"<NUM>")
    text=text.str.replace(not_allowed,"")
    return text
X=clean_text(X)
X_test=clean_text(X_test)
X_train, X_crossval, y_trains, y_crossvals = train_test_split(X, ys, test_size=0.3, random_state=20180301)

vectorizer = text.CountVectorizer()
vectorizer = text.TfidfVectorizer(max_features=1000, max_df=0.05)
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_crossval=vectorizer.transform(X_crossval)
X_test=vectorizer.transform(X_test)
model = LinearSVC()

for label in labels:
    y_train=y_trains[label]
    y_crossval=y_crossvals[label]
    model.fit(X_train, y_train)
    yh_train = model.predict(X_train)
    yh_crossval = model.predict(X_crossval)
    print(label)
    print(classification_report(y_crossval, yh_crossval))
    yh_test=model.predict(X_test)
    test_data[label]=yh_test
my_submission = test_data
my_submission.drop('comment_text',axis=1,inplace=True)
my_submission.to_csv('submission.csv', index=False)