# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Python version

import sys

print('Python: {}'.format(sys.version))

# scipy

import scipy

print('scipy: {}'.format(scipy.__version__))

# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# pandas

import pandas

print('pandas: {}'.format(pandas.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
from pandas import read_csv

from pandas.plotting import scatter_matrix 

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
url = "/kaggle/input/iris/Iris.csv" 



dataset= read_csv(url)

dataset
dataset.iloc[:,0]
# Remove first column - Id

dataset = dataset.drop(['Id'], axis=1)

# head

print(dataset.head(20))
# shape



print("This dataset have {} columns and {} rows".format(dataset.shape[0], dataset.shape[1]))
# describe

print(dataset.describe())

#class distribution





print(dataset.groupby('Species').size())
# histograms

dataset.hist()

pyplot.show()
fig=plt.figure(figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

plt.show()
# scatter plot matrix

scatter_matrix(dataset)

plt.show()
# Split-out validation dataset

array = dataset.values

X = array[:,0:4]

y = array[:,4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Spot Check Algorithms

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

	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

	results.append(cv_results)

	names.append(name)

	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))