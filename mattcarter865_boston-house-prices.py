from sys import version as py_version

from scipy import __version__ as sp_version

from numpy import __version__ as np_version

from pandas import __version__ as pd_version

from matplotlib import __version__ as mplib_version

from sklearn import __version__ as skl_version



print('python: {}'.format(py_version))

print('scipy: {}'.format(sp_version))

print('numpy: {}'.format(np_version))

print('pandas: {}'.format(pd_version))

print('matplotlib: {}'.format(mplib_version))

print('sklearn: {}'.format(skl_version))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

from pandas.tools.plotting import scatter_matrix

from sklearn.datasets import load_boston

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor

from sklearn.metrics import mean_squared_error
# Load dataset

#filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'

names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

#dataset = pd.read_csv(filename, delim_whitespace=True, names=names)

boston = load_boston()

dataset = pd.DataFrame(data=np.c_[boston['data'], boston['target']], columns=names )
# shape

dataset.shape
# types

dataset.dtypes
# peek at data

dataset.head(20)
# descriptions

pd.set_option('precision', 1)

dataset.describe()
# correlation

pd.set_option('precision',2)

dataset.corr(method='pearson')
# histograms

dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,6))

pyplot.show()
# density

dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False, fontsize=1, figsize=(12,6))

pyplot.show()
# box and whisker

dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8, figsize=(12,10))

pyplot.show()
# scatter plot matrix

scatter_matrix(dataset, figsize=(12,12))

pyplot.show()
# correlation matrix

fig = pyplot.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')

fig.colorbar(cax)

ticks = np.arange(0,14,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

fig.set_size_inches(10,10)

pyplot.show()
# validation dataset

array = dataset.values

X = array[:,0:-1]

Y = array[:,-1]

validation_size = 0.2

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# test options and evaluation metric

num_folds = 10

seed = 7

scoring = 'neg_mean_squared_error' # 0 is best
# spot check algorithms

models = []

# Linear algorithms

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('EN', ElasticNet()))

# Nonlinear algorithms

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))
# evaluate each model 

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# compare algorithms

fig = pyplot.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

fig.set_size_inches(8,6)

pyplot.show()
# standardized the dataset

pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))

pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()), ('SVR', SVR())])))
results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# compare scaled algorithms

fig = pyplot.figure()

fig.suptitle('Scaled Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

fig.set_size_inches(8,6)

pyplot.show()
# KNN algorithm tuning

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])

param_grid = dict(n_neighbors=k_values)

model = KNeighborsRegressor()

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

ranks = grid_result.cv_results_['rank_test_score']

for mean, stdev, param, rank in zip(means, stds, params, ranks):

    print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))
# ensembles

ensembles = []

# Boosting methods

ensembles.append(('ScaledAB', Pipeline([('Scalar', StandardScaler()), ('AB', AdaBoostRegressor())])))

ensembles.append(('ScaledGBM', Pipeline([('Scalar', StandardScaler()), ('GBM', GradientBoostingRegressor())])))

# Bagging methods

ensembles.append(('ScaledRF', Pipeline([('Scalar', StandardScaler()), ('RF', RandomForestRegressor())])))

ensembles.append(('ScaledET', Pipeline([('Scalar', StandardScaler()), ('ET', ExtraTreesRegressor())])))
results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# compare ensemble algorithms

fig = pyplot.figure()

fig.suptitle('Scaled Ensemble Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

fig.set_size_inches(8,6)

pyplot.show()
# tune scaled GBM

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

param_grid = {'n_estimators': np.array([50, 100, 150, 200, 250, 300, 350, 400])}

model = GradientBoostingRegressor(random_state=seed)

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

ranks = grid_result.cv_results_['rank_test_score']

for mean, stdev, param, rank in zip(means, stds, params, ranks):

    print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))
# prepare the model

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model = GradientBoostingRegressor(random_state=seed, n_estimators=250)

model.fit(rescaledX, Y_train)
# apply to our validation set to double check against over-fitting

rescaledValidationX = scaler.transform(X_validation)

predictions = model.predict(rescaledValidationX)

mean_squared_error(Y_validation, predictions)