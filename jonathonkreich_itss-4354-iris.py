import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Load libraries

from __future__ import print_function

import pandas

import matplotlib

matplotlib.use('PDF')

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

%matplotlib inline

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



import warnings

warnings.filterwarnings('ignore')
# Check the versions of libraries

print("LIBRARY VERSIONS:\n")

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



# Load dataset

url = "../input/iris(1).data"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = pandas.read_csv(url, names=names)

# shape

print("DATASET SHAPE =", dataset.shape)





# head

print("FIRST 20 ROWS OF DATA =",'\n')

print(dataset.head(20),'\n')
# descriptions

print("ATTRIBUTE SUMMARY =",'\n')

print(dataset.describe(),'\n')

# class distribution

print("CLASS DISTRIBUTION =",'\n')

print(dataset.groupby('class').size(),'\n')

# box and whisker plots

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

plt.savefig('iris1')



# histogram

dataset.hist()

plt.savefig('iris2')

# scatter plot matrix

scatter_matrix(dataset)

plt.savefig('iris3')

# Split-out validation dataset

array = dataset.values

X = array[:,0:4]

Y = array[:,4]

validation_size = 0.20

#set validation size & seed 

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



# Test options and evaluation metric

seed = 7

scoring = 'accuracy'



# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

print('MODEL ESTIMATED ACCURACY SCORES:\n')

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



# Compare Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.savefig('iris4')

#

# Make predictions on validation dataset

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

predictions = knn.predict(X_validation)

print("ACCURACY SCORE:", accuracy_score(Y_validation, predictions))


print("CONFUSION MATRIX:\n", confusion_matrix(Y_validation, predictions))

print("CLASSIFICATION REPORT:\n", classification_report(Y_validation, predictions))