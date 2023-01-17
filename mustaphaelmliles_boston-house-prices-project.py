# 1. Prepare Problem

# 1.a) Load libraries

import numpy as np

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

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# 1.b) Load dataset

filename = '/kaggle/input/boston-house-prices/housing.csv'

names = ['CRIM', 'ZN' , 'INDUS' , 'CHAS' , 'NOX' , 'RM' , 'AGE' , 'DIS' , 'RAD' , 'TAX' , 'PTRATIO' , 'B' , 'LSTAT' , 'MEDV' ]

dataset = read_csv(filename, delim_whitespace=True, names=names)
# 2. Summarize Data

# shape, type & head

set_option('display.width', 160)

set_option('precision', 6)

print('dimension :', dataset.shape,'\nType :\n', dataset.dtypes,'\nHead :\n', dataset.head(20))

# 2.a) Descriptive statistics

# summarizing the distribution of each attribute

set_option('precision', 3)

print('Statistics :\n', dataset.describe())
# correlation between attributes

print('Correlations :\n', dataset.corr(method='pearson'))

# many attributes have a strong correlation
# 2.b) Data visualizations

# histogram of individual attributes

dataset.hist(sharex = False, sharey = False,xlabelsize=1,ylabelsize=1,figsize=(18,12))

plt.show()
# density

dataset.plot(kind='density', subplots=True,  layout=(4,4),figsize=(18,12), sharex=False, legend=True,fontsize=1)

plt.show()
# box and whisker plots

dataset.plot(kind= 'box', subplots=True, layout=(4,4), sharex=False, sharey=False,fontsize=8,figsize=(18,12))

plt.show()
# visualizations of the interactions between variables : scatter matrix

scatter_matrix(dataset,figsize=(18,12))

plt.show()
# correlation matrix

fig = plt.figure()

ax = fig.add_axes([0,0,2.5,2.5])

cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation= 'none',cmap ='coolwarm')

fig.colorbar(cax)

ticks = np.arange(0,14,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

plt.show()
# 3. Evaluate Algorithms

# Split-out validation dataset

array = dataset.values

X = array[:,0:13]

Y = array[:,13]

test_size = 0.20

seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=test_size, random_state=seed)

# Standarize the data & spot-Check Algorithms

# using pipelines to avoid data leakage when we transform the data

pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))

pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

# Test options and evaluation metric

num_folds = 10

seed = 7

scoring = 'neg_mean_squared_error'

# evaluate each model in turn

results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

    

# Compare Algorithms

fig2 = plt.figure()

ax2 = fig2.add_axes([0,0,2,2])

ax2.boxplot(results, labels=names, showmeans=True, meanline=True, meanprops = dict(linestyle='--', linewidth=2.5, color='green'))

ax2.yaxis.grid(True)

ax2.set_title('Algorithm Comparison')

plt.show()
# 4. Improve Accuracy

# a) Algorithm Tuning : iterate on the nbr of neighbors 

# KNN Algorithm tuning

k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])

param_grid = dict(KNN__n_neighbors = k_values) # tunning parameter : n_neighbors

model = Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])

kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))   
# b) Ensembles

seed2 = 8

ensembles = []

ensembles.append(('ScaledAB',Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor(random_state=seed2))])))

ensembles.append(('ScaledGBM',Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor(random_state=seed2))])))

ensembles.append(('ScaledRF',Pipeline([('Scaler', StandardScaler()),('RF',RandomForestRegressor(random_state=seed2))])))

ensembles.append(('ScaledET',Pipeline([('Scaler', StandardScaler()),('ET',ExtraTreesRegressor(random_state=seed2))])))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

    

# Compare Ensemble Algorithms

fig3 = plt.figure()

ax3 = fig3.add_axes([0,0,2,2])

ax3.boxplot(results, labels=names, showmeans=True, meanline=True, meanprops = dict(linestyle='--', linewidth=2.5, color='green'))

ax3.yaxis.grid(True)

ax3.set_title('Scaled Ensemble Algorithm Comparison')

plt.show()
# ET Algorithm tuning

param_grid = dict(ET__n_estimators = np.array([50,60,80,100,150,200,250,300])) # tunning parameter : n_estimators

model = Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor(random_state=seed2))])

kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.metrics import r2_score

# 5. Finalize Model

# a) Predictions on validation dataset

# prepare the model : training the model on the entire training dataset

sc = StandardScaler()

rescaledX = sc.fit_transform(X_train)

model = ExtraTreesRegressor(random_state=seed2, n_estimators=250)

model.fit(rescaledX, Y_train)

# transform the validation dataset

rescaledTestX = sc.transform(X_test)

predictions = model.predict(rescaledTestX)

print('- mean squared error: {}, r-squared: {}' .format(-mean_squared_error(Y_test, predictions), r2_score(Y_test, predictions)))