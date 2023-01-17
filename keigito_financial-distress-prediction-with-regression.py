import numpy as np
import pandas as pd
from numpy import arange
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
%matplotlib inline
pd.set_option('display.max_rows', 50) # set the max_rows to a desired number to see all column properties
pd.set_option('display.max_columns', None) # set this to None to display all columns
figsize = [[],[]]
figsize[0] = 24
figsize[1] = 36
plt.rcParams['figure.figsize'] = figsize
filename = '../input/Financial Distress.csv'
dataset_raw = read_csv(filename)
names = list(dataset_raw.columns.values)
print(dataset_raw.head())
dataset_raw.shape
# Removing columns that will not help the model's accuracy
DropColumns = ['Company', 'Time', 'x80']
dataset = dataset_raw.copy()
dataset.drop(DropColumns, axis=1, inplace=True)
print(dataset.head(10))
print(dataset.shape)
# Checking data types
dataset.dtypes
# Checking the general summary of each columns
print(dataset.describe())
# Visualizing the data distribution
figsize[0] = 24
figsize[1] = 36
plt.rcParams['figure.figsize'] = figsize
dataset.hist(sharex=False, sharey=False, xlabelsize=12, ylabelsize=12)
plt.show()
# Somewhat redundant, but checking for a skewness of data
figsize[0] = 24
figsize[1] = 100
plt.rcParams['figure.figsize'] = figsize
dataset.plot(kind='density', subplots=True, layout=(23,4), sharex=False, sharey=False, fontsize=12)
plt.show()
# Another data distribution visualization
dataset.plot(kind='box', subplots=True, layout=(23,4), sharex=False, sharey=False, fontsize=12)
plt.show()
# Visualizing the correlations
figsize[0] = 100
figsize[1] = 70
plt.rcParams['figure.figsize'] = figsize
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,86,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
# Splitting the dataset into validation set (30% [a requirement by the stakeholder]
array = dataset.values
X = array[:, 1:82]
Y = array[:, 0]
validation_size = 0.30
seed = 3
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
num_folds = 10
scoring = 'neg_mean_squared_error'
# Initial test to identify high performing algorithms
models = []
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
InitialResults = []
InitialNames = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    InitialResults.append(cv_results)
    InitialNames.append(name)
    message = '{0}: {1} ({2})'.format(name, cv_results.mean(), cv_results.std())
    print(message)
# Visualizing the results
figsize[0] = 18
figsize[1] = 12
plt.rcParams['figure.figsize'] = figsize
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(InitialResults)
ax.set_xticklabels(InitialNames)
plt.show()
# Standardizing the data
pipelines = []
# pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('CART', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('ScaledCART', StandardScaler()), ('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()), ('SVR', SVR())])))
ScaledResults = []
ScaledNames = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    ScaledResults.append(cv_results)
    ScaledNames.append(name)
    message = '{0}: {1} ({2})'.format(name, cv_results.mean(), cv_results.std())
    print(message)
# Visualizing the results from the standardized dataset
fig = plt.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(ScaledResults)
ax.set_xticklabels(ScaledNames)
plt.show()
# Ensemble methods
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])))

EnsemblesResults = []
EnsemblesNames = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    EnsemblesResults.append(cv_results)
    EnsemblesNames.append(name)
    message = '{0}: {1} ({2})'.format(name, cv_results.mean(), cv_results.std())
    print(message)
# Visualize the results from the ensemble algorithms
fig = plt.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(EnsemblesResults)
ax.set_xticklabels(EnsemblesNames)
plt.show()
# Scaling the data for optimization
scaler = StandardScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
# Tuning parameters for GBR. Initially only one parameter will be optimized. The results from here will be used to narrow down the search space. 
model = GradientBoostingRegressor()
max_depth = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80]
min_samples_split = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80]
min_samples_leaf = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80]
max_leaf_nodes = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80]
param_grid = dict(max_depth = max_depth)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX_train, Y_train)
print('max_depth Best: {0} using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with :{2}'.format(mean, stdev, param))
print('\n')
param_grid = dict(min_samples_split = min_samples_split)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX_train, Y_train)
print('min_samples_split Best: {0} using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with :{2}'.format(mean, stdev, param))
print('\n')
param_grid = dict(min_samples_leaf = min_samples_leaf)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX_train, Y_train)
print('min_samples_leaf Best: {0} using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with :{2}'.format(mean, stdev, param))
min_samples_leaf = min_samples_leaf
print('\n')
param_grid = dict(max_leaf_nodes = max_leaf_nodes)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX_train, Y_train)
print('max_leaf_nodes Best: {0} using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with :{2}'.format(mean, stdev, param))
min_samples_leaf = min_samples_leaf
# Multi-parameter grid search
max_depth = [2, 4, 6, 8, 10]
min_samples_leaf = [20, 30, 40, 50, 60]
max_leaf_nodes = [2, 4, 6, 8, 10]
param_grid = dict(max_depth = max_depth, min_samples_leaf = min_samples_leaf, max_leaf_nodes = max_leaf_nodes)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX_train, Y_train)
print('Best: {0} using {1}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with :{2}'.format(mean, stdev, param))
# Finalizing the model
model = GradientBoostingRegressor(min_samples_split=8, max_depth=4, min_samples_leaf=40)
model.fit(rescaledX_train, Y_train)

# ... and see how the model performs against the validation set
rescaledX_validation = scaler.transform(X_validation)

predictions = model.predict(rescaledX_validation)

print(mean_squared_error(Y_validation, predictions))

# Finally, converting the Financial Distress values to a "Healthy/Distressed" results
FinalPredictions = []
Test = []
for i in predictions:
    if i > -0.5:
        FinalPredictions.append(0)
    else:
        FinalPredictions.append(1)

print(FinalPredictions)
