# Check the versions of libraries



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
# Load libraries

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
# Check Kaggle filename

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load dataset

url = "/kaggle/input/iris/Iris.csv"

#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = read_csv(url)
# shape

print(dataset.shape)
# head

print(dataset.head(20))
# descriptions

print(dataset.iloc[:,1:].describe())
# Create a dataset backup

dataset_bak = dataset
dataset.iloc[:,0]
# Remove first column - Id

dataset = dataset.drop(['Id'], axis=1)

# head

print(dataset.head(20))
# box and whisker plots

fig=plt.figure(figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

plt.show()
# histograms

dataset.hist()

plt.show()
# scatter plot matrix

scatter_matrix(dataset)

plt.show()
# Split-out validation dataset

array = dataset.values

X = array[:,0:4]

y = array[:,4]



# Split 80-20 train-test data

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Visualize first ten results of train data

X_train[1:10,]
# Visualize first ten results of validation data

X_validation[1:10,]
# Visualize first ten results of train predictor

Y_train[1:10,]
# Visualize first ten results of validation predictor

Y_validation[1:10,]
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