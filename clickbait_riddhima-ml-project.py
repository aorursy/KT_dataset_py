import sys

import numpy

import matplotlib

import pandas

import sklearn

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.metrics import classification_report, accuracy_score

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import pandas as pd

import os

import warnings

warnings.filterwarnings("ignore")



filepath = ''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filepath = os.path.join(dirname, filename)



print(filepath)
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

df = pd.read_csv(filepath, names = names)

df.replace('?', -99999, inplace = True)

print(df.axes)
df.drop(['id'], 1, inplace = True)

print(df.loc[6])

print(df.describe())
df.hist(figsize = (10, 10))

plt.show()
scatter_matrix(df, figsize = (18, 18))

plt.show()
X = np.array(df.drop(['class'], 1))

y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)



seed = 8

scoring = 'accuracy'

models = []

models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))

models.append(('SVM', SVC()))

results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits = 10, random_state = seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s : %f(%f)" %(name, cv_results.mean(), cv_results.std()) 

    print(msg)
for name, model in models:

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name)

    print(accuracy_score(y_test, predictions))

    print(classification_report(y_test, predictions))
clf = SVC()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)
clf =  KNeighborsClassifier(n_neighbors = 5)

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)