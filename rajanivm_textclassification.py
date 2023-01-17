import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import os

from sklearn import datasets

from sklearn.model_selection import train_test_split



print(os.path.exists("../input/enron-sample"))

print(os.path.exists("../input/sampledata/sample"))

#data = datasets.load_files ('../input/sampledata/sample/sample')

data = datasets.load_files('../input/enron-email-labelled-datasets/enron_with_categories/enron_with_categories')

for keys in data.keys() :

    print(keys)

print(list(data.target_names))

print(len(data.data))

X=data.data

y=data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#train, test = train_test_split(X, test_size=0.2,random_state=42)

#print(list(test))

print(len(y_train))

print(len(y_test))

print(y_train)

#print(list(y_train))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

vectortrain = vectorizer.fit_transform(X_train)

vectortrain.shape
vectortest = vectorizer.transform(X_test)

vectortest.shape
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics



clf = MultinomialNB(alpha=.01)

clf.fit(vectortrain, y_train)

MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)

pred = clf.predict(vectortest)

metrics.f1_score(y_test, pred, average='macro')

metrics.accuracy_score(y_test, pred)
metrics.confusion_matrix(y_test, pred)
print(metrics.classification_report(y_test, pred, target_names=data.target_names))
