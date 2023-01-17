# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

Psy = pd.read_csv('../input/Youtube01-Psy.csv')
Katy = pd.read_csv('../input/Youtube02-KatyPerry.csv')
Eminem = pd.read_csv('../input/Youtube04-Eminem.csv')
Shakira = pd.read_csv('../input/Youtube05-Shakira.csv')
LMFAO = pd.read_csv('../input/Youtube03-LMFAO.csv')

df_spam = pd.concat([Shakira, Eminem, Katy, Psy, LMFAO])
df_spam.drop('DATE', axis=1, inplace=True)
df_spam.head()
df_spam.shape
df_spam['CLASS'].value_counts().plot(kind='bar')
# This is to define the features and labels for the CountVectorizer
X = df_spam.CONTENT
y = df_spam.CLASS
print(X.shape)
print(y.shape)
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# import and instantiate CountVectorizer (with the default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
# examine the fitted vocabulary
vect.get_feature_names()
# transform training data into a 'document-term matrix with a single step
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix
X_train_dtm
# transform testing data into a document-term matrix
# using the transform() method
X_test_dtm = vect.transform(X_test)
X_test_dtm
# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)
# make predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)
# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)
# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class, digits=4))
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]
y_pred_prob
# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)