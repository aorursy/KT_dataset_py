# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load libraries

from numpy import arange

from numpy import array

from numpy import argmax

from matplotlib import pyplot

from pandas import read_csv

from pandas import set_option

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor

from sklearn.metrics import mean_squared_error
# Load dataset

filename = '/kaggle/input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv'

dataset = read_csv(filename)
dataset.shape
dataset.dtypes
dataset.head(10)
dataset.describe()
dataset.plot(kind = 'density', subplots = True, layout = (4,4), sharex = False, sharey = False)
# Drop Columns

dataset = dataset.drop(['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis = 1)
# Drop NaN

dataset = dataset.dropna(axis = 0)
# Convert User_Score to Float

dataset['User_Score'] = pd.to_numeric(dataset['User_Score'])
# One-Hot Encode

dataset = pd.get_dummies(dataset)
# Move Global_Sales to End of Dataset

dataset.head(5)
Y = dataset['Global_Sales'].values
dataset = dataset.drop(['Global_Sales'], axis = 1)

array = dataset.values

X = array
validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, 

                                                                random_state = seed)
# Test options and evaluation metric

num_folds = 10

seed = 7

scoring = 'neg_mean_absolute_error'
# Spot Check Algorithms

models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits = num_folds, random_state = seed)

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

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR',

    LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO',

    Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN',

    ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN',

    KNeighborsRegressor())])))
# Spot Check Scaled Algorithms

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state = seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring, error_score = numpy.nan)

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
# Tuning CART

depth = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41]

param_grid = dict(max_depth=depth)

model = DecisionTreeRegressor()

kfold = KFold(n_splits=num_folds, random_state = seed)

grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)

grid_result = grid.fit(X_train, Y_train)
# Display Scores

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
num_folds = 10

seed = 7
ensembles = []

ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))

ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))

ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestRegressor(n_estimators = 10))])))

ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor(n_estimators = 10))])))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits = num_folds, random_state = seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Scale Ensemble Algorithms

fig = pyplot.figure()

fig.suptitle('Scaled Ensemble Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
# Tune Scaled RF

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

param_grid = dict(max_depth = [10, 20, 30, 40, 50, 60])

model = RandomForestRegressor(n_estimators = 10, random_state=seed)

kfold = KFold(n_splits = num_folds, random_state = seed)

grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv=kfold, iid = True)

grid_result = grid.fit(rescaledX, Y_train)
# Display Scores

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# Tune Scaled ET

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

param_grid = dict(max_depth = [10, 20, 30, 40, 50, 60])

model = ExtraTreesRegressor(n_estimators = 10, random_state=seed)

kfold = KFold(n_splits = num_folds, random_state = seed)

grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv=kfold, iid = True)

grid_result = grid.fit(rescaledX, Y_train)
# Display Scores

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model = ExtraTreesRegressor(n_estimators = 10, max_depth = 50, random_state = seed)

model.fit(rescaledX, Y_train)
# Predict on Validation Set

rescaledValidationX = scaler.transform(X_validation)

predictions = model.predict(rescaledValidationX)

print(mean_squared_error(Y_validation, predictions))