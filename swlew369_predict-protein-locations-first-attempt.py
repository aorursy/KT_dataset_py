import pandas as pd

import seaborn as sns

import numpy as np

import time

import re

import nltk

import math

import os

import matplotlib.pyplot as plt

import sklearn.metrics

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import roc_auc_score



%matplotlib inline

# data downloaded from kaggle

# https://www.kaggle.com/lzyacht/proteinsubcellularlocalization/downloads/proteinsubcellularlocalization.zip/1

# data is from SWISS-PROT database release 42 (2003–2004)

df = pd.read_csv('../input/proteinsLocations.csv')

df.head(3)
df.shape
df.dtypes
df.label.value_counts()
df.isnull().sum()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(df.label)

df['location'] = le.transform(df['label'])
df.head(3)
df.tail()
df.dtypes
df.location.value_counts()
# select only two types of proteins: cytoplasmic and plasma membrane proteins

mask = ((df.location == 3) | (df.location == 9))

#mask = ((df.location == 3) | (df.location == 6))

data = df[mask]

data.head()
# scramble the data

data = data.sample(frac=1)

data.head(3)
data.location.value_counts()
vect_3 = CountVectorizer(min_df=1,token_pattern=r'\w{1}',ngram_range=(3, 3))



X = vect_3.fit_transform(data.sequence)

y = data.location



# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state =42)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
print(len(y_train))

print(len(y_test))
y_train.value_counts()
y_test.value_counts()
# Logistic Regression using CountVectorizer for tripeptide frequency

lr = LogisticRegression()

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

print("Score: {:.2f}".format(lr.score(X_test, y_test)))
# Try optimizing Logistic Regression model

#the grid of parameters to search over

Cs = [0.001,0.01, 0.1, 1, 10, 100]



Scores = []



for item in Cs:

    clf = LogisticRegression(C=item)

    clf.fit(X_train, y_train)

    Scores.append((clf.score(X_test, y_test)))

    

Scores
score_highest = max(Scores)

print(score_highest)

print()

print(Scores.index(score_highest))

print()

C_opt = Cs[Scores.index(score_highest)]

print(C_opt)

print()
# Logistic Regression using CountVectorizer for tripeptide frequency Optimized

lr2 = LogisticRegression(C=C_opt)

lr2.fit(X_train, y_train)

#predictions = lr2.predict(X_test)

print("Score: {:.2f}".format(lr2.score(X_test, y_test)))
vect_3 = CountVectorizer(min_df=1,token_pattern=r'\w{1}',ngram_range=(3, 3))



X = vect_3.fit_transform(df.sequence)

y = df.location



# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state =42)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
print(len(y_train))

print(len(y_test))
y_train.value_counts()
y_test.value_counts()
# Logistic Regression using CountVectorizer for tripeptide frequency

lr_all = LogisticRegression()

lr_all.fit(X_train, y_train)

predictions = lr_all.predict(X_test)

print("Score: {:.2f}".format(lr_all.score(X_test, y_test)))
# Try optimizing Logistic Regression model

#the grid of parameters to search over

Cs = [0.001,0.01, 0.1, 1, 10, 100]



Scores = []



for item in Cs:

    clf = LogisticRegression(C=item)

    clf.fit(X_train, y_train)

    Scores.append((clf.score(X_test, y_test)))

    

Scores
score_highest = max(Scores)

print(score_highest)

print()

print(Scores.index(score_highest))

print()

C_opt = Cs[Scores.index(score_highest)]

print(C_opt)

print()
# Logistic Regression using CountVectorizer for tripeptide frequency Optimized

lr2 = LogisticRegression(C=C_opt)

lr2.fit(X_train, y_train)

#predictions = lr2.predict(X_test)

print("Score: {:.2f}".format(lr2.score(X_test, y_test)))