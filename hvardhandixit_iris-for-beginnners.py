# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
data.info()
fig, ax = plt.subplots(1, 2, figsize = (20, 7))

ax = sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = data, hue = 'Species', ax = ax[0])

ax1 = sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = data, hue = 'Species')
fig, ax = plt.subplots(1, 2, figsize = (20, 7))

ax = sns.regplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = data, ax = ax[0])

ax1 = sns.regplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = data)
d = data.drop('Id', axis = 1)

fig = plt.figure(figsize = (10, 10))

ax = sns.pairplot(d, hue= 'Species')
data.drop('Id', axis = 1, inplace = True)

data.info()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
train, test = train_test_split(data, test_size = 0.3)

print(train.shape)

print(test.shape)
train_X = train.drop('Species', axis = 1)

train_y = train['Species']

test_X = test.drop('Species', axis = 1)

test_y = test['Species']

test_X.head()
model = LogisticRegression()

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print("Accuracy = {}".format(metrics.accuracy_score(prediction, test_y)))
model = svm.SVC()

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print("Accuracy = {}".format(metrics.accuracy_score(prediction, test_y)))
model = DecisionTreeClassifier()

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print("Accuracy = {}".format(metrics.accuracy_score(prediction, test_y)))
model = KNeighborsClassifier(n_neighbors=2)

model.fit(train_X, train_y)

prediction = model.predict(test_X)

print("Accuracy = {}".format(metrics.accuracy_score(prediction, test_y)))