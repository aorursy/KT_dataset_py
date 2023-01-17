import numpy as np

import pandas as pd

#import seaborn as sns

#sns.set_palette('husl')

import matplotlib.pyplot as plt

#%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



IRIS = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
IRIS.head()
IRIS.info()
IRIS.describe()
IRIS['species'].value_counts()
X = IRIS.drop(['species'], axis=1)

y = IRIS['species']

# print(X.head())

print(X.shape)

# print(y.head())

print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
def get_score(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    return model.score(X_test, y_test)
# Logistic Regression

get_score(LogisticRegression(), X_train, X_test, y_train, y_test)
# SVM

get_score(SVC(), X_train, X_test, y_train, y_test)
# Random Forest

get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test)
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X, y, cv=3)
cross_val_score(SVC(gamma='auto'), X, y, cv=3)
cross_val_score(RandomForestClassifier(n_estimators=40), X, y, cv=3)
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),X, y, cv=10)

np.average(scores1)
scores2 = cross_val_score(RandomForestClassifier(n_estimators=20), X, y, cv=10)

np.average(scores2)
scores3 = cross_val_score(RandomForestClassifier(n_estimators=30), X, y, cv=10)

np.average(scores3)
scores4 = cross_val_score(RandomForestClassifier(n_estimators=40), X, y, cv=10)

np.average(scores4)