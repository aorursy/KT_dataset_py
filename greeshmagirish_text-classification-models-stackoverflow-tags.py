import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

from bs4 import BeautifulSoup

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import classification_report
df = pd.read_csv('/kaggle/input/stack-overflow-data.csv')

df = df[pd.notnull(df['tags'])]
def clean_text(text):

    """

        text: a string

        return: modified initial string

    """

    text = BeautifulSoup(text, "lxml").text # HTML decoding

    text = text.lower() # lowercase text

    return text



df['post'] = df['post'].apply(clean_text)
df.head()
X_train, X_test, y_train, y_test = train_test_split(df['post'], df['tags'], random_state=1087, test_size=0.2)
nb = Pipeline([

                ('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', MultinomialNB()),

              ])



nb.fit(X_train, y_train)



y_pred = nb.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
svm = Pipeline([

                ('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', LinearSVC()),

              ])



svm.fit(X_train, y_train)



y_pred = svm.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
svm_sgd = Pipeline([

                ('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=10, random_state=42))

              ])



svm_sgd.fit(X_train, y_train)



y_pred = svm_sgd.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
lgclf = Pipeline([

                ('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', LogisticRegression(random_state=0)),

              ])



lgclf.fit(X_train, y_train)



y_pred = lgclf.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))
rfc = Pipeline([

                ('vect', CountVectorizer()),

                ('tfidf', TfidfTransformer()),

                ('clf', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)),

              ])



rfc.fit(X_train, y_train)



y_pred = rfc.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))