import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/kyphosis.csv')

df.head()
df.info()
df.describe()
#handling categorical data

kyphosis = pd.get_dummies(df['Kyphosis'], drop_first=True)

df = pd.concat([df, kyphosis], axis=1).drop(['Kyphosis'], axis=1)

df.head()
sns.pairplot(df, hue='present')
#splitting train and test data

from sklearn.model_selection import train_test_split

X = df.drop('present', axis=1)

y = df['present']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#using descision tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

prediction = dtree.predict(X_test)

#checking performance of the model

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))

#using random forests

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

predictionRFC = rfc.predict(X_test)

print(classification_report(y_test, predictionRFC))

print(confusion_matrix(y_test, predictionRFC))