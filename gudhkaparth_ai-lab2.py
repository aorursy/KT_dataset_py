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
import pandas as pd

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
dataset = pd.read_csv("../input/IRIS.csv")
print(dataset.head(10))
print(dataset.describe())
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

plt.show()
dataset.hist()

plt.show
scatter_matrix(dataset)

plt.show()
array = dataset.values

x = array[:,0:4]
print(x)
y = array[:,4]

print(y)
validation_size=0.20
seed= 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)
seed = 7

scoring = 'accuracy'
models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn

results = []

names = []

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=seed)

	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)
fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

predictions = knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))