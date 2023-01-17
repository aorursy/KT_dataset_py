import numpy as np

import pandas as pd

import nltk
data=pd.read_csv('../input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/imdb_labelled.txt',sep='\t',header=None)

data.head()
data.columns=['text','label']

data.head()
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(stop_words='english')

X_train_counts = count_vect.fit_transform(data.text)

X_train_counts.shape
#TFIDF -> Pembobotan [0-8]



from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer()

X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape
y=data.label
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X_train_tfidf,y,test_size=0.33,random_state=42)
from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()

nb.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

y_pred=nb.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier()

tree.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

y_pred=tree.predict(X_test)

print(accuracy_score(y_pred,y_test))
from sklearn.svm import SVC

svc=SVC()

svc.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

y_pred=svc.predict(X_test)

print(accuracy_score(y_pred,y_test))