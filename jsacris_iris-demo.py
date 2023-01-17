# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn import datasets
import matplotlib.pyplot as plt

iris_ds = datasets.load_iris()
# X = iris_ds.data[:, :2]
X = iris_ds.data
y = iris_ds.target

# print(iris_ds.target_names)
# print(iris_ds.feature_names)

my_df = pd.DataFrame(columns=iris_ds.feature_names)

for idx in range(len(X)):
    my_sample = X[idx]
    my_df.loc[idx] = my_sample

my_df.tail(10)
fig, ax1 = plt.subplots()
ax1.scatter( X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')

ax1.set_xlabel(iris_ds.feature_names[0])
ax1.set_ylabel(iris_ds.feature_names[1])

plt.show()
fig, ax1 = plt.subplots()
ax1.scatter( X[:, 0], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k')

ax1.set_xlabel(iris_ds.feature_names[0])
ax1.set_ylabel(iris_ds.feature_names[2])

plt.show()
fig, ax1 = plt.subplots()
ax1.scatter( X[:, 0], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k')

ax1.set_xlabel(iris_ds.feature_names[0])
ax1.set_ylabel(iris_ds.feature_names[3])

plt.show()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# At this point we have a train set (X_train, y_train) and a test set (X_test, y_test)

# Print the lenght of each set
print(X_train.shape)
print(X_test.shape)

# create a classifier
my_clf = KNeighborsClassifier(1)

# Fit a classifier
my_clf.fit(X_train, y_train)
print(my_clf)
y_predicted = my_clf.predict(X_test)
print(y_predicted)
print(y_test)

# Quick and dirty comparison
print(y_predicted - y_test)