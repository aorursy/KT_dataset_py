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
# Load Libraries

import numpy

import seaborn as sns

from matplotlib import pyplot

from pandas import read_csv, set_option

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.feature_selection import RFE

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
# Load dataset

filename = '/kaggle/input/Cryotherapy.csv'

dataset = read_csv(filename)
# Descriptive Statistics

dataset.shape
dataset.dtypes
dataset.head(20)
dataset.describe()
dataset.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize = 1)

pyplot.show()
dataset.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False, legend = False, 

             fontsize = 1)

pyplot.show()
fig = pyplot.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(dataset.corr(), vmin = -1, vmax = 1, interpolation = 'none')

fig.colorbar(cax)

pyplot.show()
# Split Validation dataset

array = dataset.values

X = array[:,0:6]

Y = array[:,6]

validation_size = 0.33

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, 

                                                                random_state = seed)
# Test options and Evaluation Metric

num_folds = 10

seed = 7

scoring = 'accuracy'
# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver = 'liblinear')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)

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
# Standardization

pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression(solver='liblinear'))])))

pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))

pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))

pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()), ('SVM', SVC(gamma='auto'))])))

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits = num_folds, random_state = seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Scaled Algorithms

fig = pyplot.figure()

fig.suptitle('Scaled Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
# Algorithm Tuning

# LDA

solver = ['svd', 'lsqr', 'eigen']

param_grid = dict(solver=solver)

model = LinearDiscriminantAnalysis()

kfold = KFold(n_splits = num_folds, random_state = seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring = scoring, cv=kfold, iid=True)

grid_result = grid.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# CART

depth = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

param_grid = dict(max_depth = depth)

model = DecisionTreeClassifier()

kfold = KFold(n_splits = num_folds, random_state = seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring = scoring, cv= kfold, iid = True)

grid_result = grid.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# Ensemble Algorithms

ensembles = []

ensembles.append(('AB', AdaBoostClassifier()))

ensembles.append(('GBM', GradientBoostingClassifier()))

ensembles.append(('RF', RandomForestClassifier(n_estimators = 10)))

ensembles.append(('ET', ExtraTreesClassifier(n_estimators = 10)))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits = num_folds, random_state = seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Ensemble Algorithms

fig = pyplot.figure()

fig.suptitle("Ensemble Algorithm Comparison")

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
# Finalize Model

model = DecisionTreeClassifier(max_depth = 3)

model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))