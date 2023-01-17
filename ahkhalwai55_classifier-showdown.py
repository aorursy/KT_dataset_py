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
# Load libraries

import pandas as pd

import numpy

from matplotlib import pyplot

from pandas import read_csv

from pandas import set_option

from pandas.tools.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")
# Load dataset

train = read_csv('../input/train.csv')

test = read_csv('../input/test.csv')
# First 5 row

train.head()
# First 5 row

test.head()
# shape

train.shape
# shape

test.shape
# types

train.dtypes
# types

test.dtypes
# Ignore the first column (id) and the second column (species) (pandas.DataFrame)

x = train.iloc[:, 2:].values
x[0]
# Convert the species to category type

y = train['species'].astype('category')
y
# Get the corresponding categories list for species (numpy.ndarray)

y = y.cat.codes.as_matrix()
y
# Split-out validation dataset

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(x, y,

test_size=validation_size, random_state=seed)
# Test options and evaluation metric

num_folds = 10

seed = 7

scoring = 'accuracy'
# Spot-Check Algorithms

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Algorithms

fig = pyplot.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
# Standardize the dataset

pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression())])))

pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsClassifier())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeClassifier())])))

pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',GaussianNB())])))

pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Algorithms

fig = pyplot.figure()

fig.suptitle('Scaled Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
# Tune scaled SVM

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

param_grid = dict(C=c_values, kernel=kernel_values)

model = SVC()

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# ensembles

ensembles = []

ensembles.append(('AB', AdaBoostClassifier()))

ensembles.append(('GBM', GradientBoostingClassifier()))

ensembles.append(('RF', RandomForestClassifier()))

ensembles.append(('ET', ExtraTreesClassifier()))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# prepare the model

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model = SVC(C= 0.1, kernel= 'linear',probability=True)

model.fit(rescaledX, Y_train)
# estimate accuracy on validation dataset

rescaledValidationX = scaler.transform(X_validation)

predictions = model.predict(rescaledValidationX)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))
submission_path = '../input/sample_submission.csv'

submission_data = pd.read_csv(submission_path)
submission_data
test = pd.read_csv('../input/test.csv')

test_ids = test.pop('id')

x_test = test.values

scaler = StandardScaler().fit(x_test)

x_test = scaler.transform(x_test)
test_y = model.predict_proba(x_test)
submission_data.iloc[:, 1:] = test_y

submission_data.tail()
submission_output = 'submit.csv'

f = open(submission_output, 'w')

f.write(pd.DataFrame(submission_data).to_csv(index = False))

f.close()