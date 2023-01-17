#coding: utf8



import numpy as np

import pandas as pd



from sklearn.naive_bayes import MultinomialNB

from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/spam.csv", encoding="ISO-8859-1")

data = df.as_matrix()

df.head()
# Note: I think there's an error in the CSV formatting that's causing

# a few unnecessary columns to be added

df.describe()
print(data.shape)

X = data[:, 1]

Y = data[:, :1]

print(X.shape, Y.shape)

CV = CountVectorizer().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(CV, Y, test_size=0.33, random_state=42)

print(X_train.shape, X_test.shape)
model = MultinomialNB()

model.fit(X_train, y_train.ravel())

print( "Classificiation rate for NB:", model.score(X_test, y_test))

preds = model.predict(X_test)

print(classification_report(y_test, preds))
model = RandomForestClassifier()

model.fit(X_train, y_train.ravel())

print( "Classificiation rate for RandomForest:", model.score(X_test, y_test))

preds = model.predict(X_test)

print(classification_report(y_test, preds))
from sklearn.neural_network import MLPClassifier

model = MLPClassifier()

model.fit(X_train, y_train.ravel())

print( "Classificiation rate for MLPClassifier:", model.score(X_test, y_test))

preds = model.predict(X_test)

print(classification_report(y_test, preds))