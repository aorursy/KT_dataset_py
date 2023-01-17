# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import unicode_literals

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import re

import string



from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer



from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score



from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file_path = "../input/Womens Clothing E-Commerce Reviews.csv"

df = pd.read_csv(file_path,index_col = 0 )

df.info()

df.groupby('Recommended IND').describe()
# df.info()

# print(df.shape) # (23486, 10)

df = df.dropna(subset=['Review Text'])

df.isnull().sum() # no more null Review Text

rec = df.filter(['Review Text','Recommended IND'])

rec.columns = ['text','target']
stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# remove all tokens that are stop words or punctuations and perform lemmatization

def prep_clean(text):

    text = text.lower()

    text = re.sub(r'\d+','',text)

    tokens = word_tokenize(text)

    words = [token for token in tokens if not token in stop_words]

    words = [stemmer.stem(word) for word in words]

    words = [word for word in words if not word in string.punctuation]

    words = [word for word in words if len(word) > 1]

    return words
X_train, X_test, y_train, y_test = train_test_split(rec['text'], rec['target'], test_size=0.2)
# build the model and test its accuracy

def model(mod, name, X_train, X_test, y_train, y_test):

    mod.fit(X_train, y_train)

    print(name)

    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv=5)

    predictions = cross_val_predict(mod, X_train, y_train, cv=5)

    print("Accuracy: ", round(acc.mean(),3))

    cm = confusion_matrix(predictions, y_train)

    print("Confusion Matrix: \n", cm)

    print("Classification Report: \n", classification_report(predictions, y_train))

    print("--------")

    print(predictions[:10])

    print(y_train[:10])

# the model and all the preprocessing steps

def pipeline(bow, tfidf, model):

    return Pipeline([('bow', bow),

               ('tfidf', tfidf),

               ('classifier', model),

              ])
mnb = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), MultinomialNB())

mnb = model(mnb, "Multinomial Naive Bayes", X_train, X_test, y_train, y_test)



log = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LogisticRegression(solver='lbfgs'))

log = model(log, "Logistic Regression", X_train, X_test, y_train, y_test)



svc = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LinearSVC())

svc = model(svc, "Linear SVC", X_train, X_test, y_train, y_test)
rec["concat"] = df["Title"].fillna('') + " "+ df["Review Text"]

X_train, X_test, y_train, y_test = train_test_split(rec['concat'], rec['target'], test_size=0.2)

mnb = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), MultinomialNB())

mnb = model(mnb, "Multinomial Naive Bayes", X_train, X_test, y_train, y_test)



log = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LogisticRegression(solver='lbfgs'))

log = model(log, "Logistic Regression", X_train, X_test, y_train, y_test)



svc = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LinearSVC())

svc = model(svc, "Linear SVC", X_train, X_test, y_train, y_test)



# rfc = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LinearSVC())

# rfc = model(rfc, "Random Forest Classifier", X_train, X_test, y_train, y_test)
