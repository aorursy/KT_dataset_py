import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/heart.csv')
data.head()
x = data.iloc[:,:-1].values

y = data.iloc[:,13].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
x_train[1:5]
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
x_train
plt.figure(figsize=(15,8))

sns.heatmap(data.corr(),annot=True)
data.info()
data.describe()
'''Naive Bayes'''
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(x_train,y_train)

y_pref=clf.predict(x_test)

clf.score(x_test, y_test)*100
"""Logistic Regression"""
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(x_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
y_pred
classifier.score(x_test, y_test)*100
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
"""KNN"""
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 6)

clf.fit(x_train,y_train)

y_pref=clf.predict(x_test)

clf.score(x_test, y_test)*100
"""Decision Tree Classifier"""
from sklearn.tree import DecisionTreeClassifier

clf1 = DecisionTreeClassifier()

clf1.fit(x_train,y_train)

y_pref=clf1.predict(x_test)

clf1.score(x_test, y_test)*100
"""Random forest"""
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=13)

clf.fit(x_train,y_train)

y_pref=clf.predict(x_test)

clf.score(x_test, y_test)*100
"""Support Vector Classifier"""
from sklearn.svm import SVC

clf2 = SVC()

clf2.fit(x_train,y_train)

y_pref=clf.predict(x_test)

clf2.score(x_test, y_test)*100