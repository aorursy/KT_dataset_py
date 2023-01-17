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
import matplotlib.pyplot as plt

import sklearn

import scipy



ir = pd.read_csv("../input/Iris.csv")
print(ir.shape)

print(ir.head(5))

ir1 = ir.drop("Id",1)

ir1.head(5)



ir1.describe()
ir1.plot(kind = 'box', subplots = True, layout = (2,2), sharey = False,figsize =(6,6))

plt.show()
ir1.hist(figsize= (6,6))

plt.show()
ir1.plot(kind = "density", subplots = True, sharey = False, figsize = (6,6))

plt.show()

species_table = pd.crosstab(index = ir1["Species"], columns = "count")

species_table
species_table.plot(kind = "bar" )

plt.show()
ir1.plot(kind = "bar")

plt.show()

ir1.plot(kind = "bar", stacked = True)

plt.show()
from pandas.plotting import scatter_matrix



scatter_matrix(ir1)

plt.show()
ir1.plot(subplots = True,  figsize = (6,6))

plt.show()
from sklearn import model_selection

array = ir1.values

X = array[:, 0:4]

Y = array[:,4]

validation_size = 0.2

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



scoring = 'accuracy'

models = []

models.append(('LR', LogisticRegression()))

models.append(('LD', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

results = []

names =[]

for name, model in models:

    kfold = model_selection.KFold(n_splits= 10, random_state = seed)

    cv_results = model_selection.cross_val_score(model,X_train, Y_train, cv = kfold, scoring = scoring)

    results.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



# Compare Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Make predictions on validation dataset

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))