# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv')

iris.info()

iris.head(5)
# setosa flowers are the most seperable species

sns.pairplot(iris, hue='Species')
# get all flowers with setosa type from the dataser

setosa = iris[iris['Species'] == 'Iris-setosa']
# compare width and length for setosa flowers

sns.kdeplot(setosa['SepalWidthCm'], setosa['SepalLengthCm'])
X = iris.iloc[:, 1:5]

Y = iris.iloc[:,5]
# dataset looks balanced

Y.value_counts()
# scale the data

sc = StandardScaler()

X = sc.fit_transform(X)
print(len(X))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
svc_model = SVC()

svc_model.fit(X_train, y_train)



predictions = svc_model.predict(X_test)



print(confusion_matrix(y_test, predictions))



print(classification_report(y_test, predictions))
# use gridsearch to find better parameters

param_grid = {'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001]}



grid = GridSearchCV(SVC(), param_grid, verbose=2)

grid.fit(X_train, y_train)
grid_preds = grid.predict(X_test)

print(classification_report(y_test, grid_preds))
dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

print(classification_report(y_test, predictions))
# here we decide to use 20 different decision trees together

rfc = RandomForestClassifier(n_estimators=45)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(classification_report(y_test, rfc_pred))