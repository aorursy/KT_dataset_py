import scipy

import numpy as np

import matplotlib

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.svm import SVC
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

df = pd.read_csv(url, names=names)

df.head()
%matplotlib inline

df.hist()

plt.show()

def plot_corr(df):

    f = plt.figure(figsize=(8, 8))

    corr = df.corr()

    plt.matshow(corr, fignum=f.number)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)

    



plot_corr(df)

df.corr()
from sklearn.model_selection import train_test_split



feature_col_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

predicted_class_names = ['class']



X = df[feature_col_names].values

Y = df[predicted_class_names].values

split_test_size = 0.25



X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=split_test_size, random_state = 42)
models = [('LogisticRegression', LogisticRegression()), ('KNeighborsClassifier', KNeighborsClassifier()), ('SVM', SVC()),

         ('XGBClassifier', XGBClassifier())]



results = []

names = []



for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state = 42)

    cv_results = model_selection.cross_val_score(model, X_train, y_train.ravel(), cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
for name, model in models:

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name)

    print(accuracy_score(y_test, predictions))

    print(classification_report(y_test, predictions))