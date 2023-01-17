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
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from matplotlib import pyplot

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor

from sklearn.metrics import mean_squared_error
seed = 7

np.random.seed(seed)

filename = '/kaggle/input/abalone-uci/abalone_original.csv'

dataset = pd.read_csv(filename)

print(dataset.shape)

print(dataset.head(10))

dataset.isnull().sum()

print(dataset.dtypes)
encoder = LabelEncoder()

encoder.fit(dataset['sex'])

dataset['sex'] = encoder.transform(dataset['sex'])



dataset['age'] = dataset['rings'] + 1.5

dataset = dataset.drop('rings', axis=1)

dataset.head(10)
pd.set_option('precision', 1)

print(dataset.describe())

pd.set_option('precision', 2)

print(dataset.corr(method='pearson'))
# Data visualization

dataset.hist(sharex = False, sharey = False, xlabelsize=1, ylabelsize=1)



# Density 

dataset.plot(kind='density', subplots=True, layout=(9,9), sharex =False, legend=False, fontsize=1)

pyplot.show()



# Box plot

dataset.plot(kind='box', subplots=True, layout=(9,9), sharex=False, sharey=False, legend=False, fontsize=1)

pyplot.show()
# Multimodal data visualizations

# scatter plot matrix

scatter_matrix(dataset)

pyplot.show()
# correlation matrix

ax = pyplot.figure().add_subplot(111)

cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')

pyplot.figure().colorbar(cax)

#pyplot.show()
# split out validation dataset

array = dataset.values

x = array[:,0:8]

y = array[:,8]

print(dataset.head(10))

print(y)

validation_size=0.3

seed = 7

x_train, x_validation, y_train, y_validation = train_test_split(x,y,test_size=validation_size,random_state=seed)

# Evaluate algorithms

num_folds=10

scoring = 'neg_mean_squared_error'
# spot check algorithms

models = []

models.append(('LR', LinearRegression()))

models.append(('Lasso', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))
# Evaluate each model

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{}: {}, {}".format(name, cv_results.mean(), cv_results.std()))
# compare algorithms

fig = pyplot.figure()

fig.suptitle('Algorithm comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
# Evaluate algorithms: Standardization

pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))

pipelines.append(('ScaledLasso', Pipeline([('Scaler', StandardScaler()),('Lasso', Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))

pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{}: {}, {}".format(name, cv_results.mean(), cv_results.std()))
# improve results with SVR

scaler = StandardScaler().fit(x_train)

rescaledx = scaler.transform(x_train)

c_values = [0.1, 0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,2.0]

kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

param_grid = dict(C=c_values, kernel=kernel_values)

model = SVR()

kfold = KFold(n_splits=10, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(rescaledx, y_train)

print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

ensembles = []

ensembles.append(('AB', AdaBoostRegressor()))

ensembles.append(('GBM', GradientBoostingRegressor()))

ensembles.append(('RF', RandomForestRegressor()))

ensembles.append(('ET', ExtraTreesRegressor()))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=10, random_state=seed)

    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{}: {}, {}".format(name, cv_results.mean(), cv_results.std()))
# Finalize model

# prepare model

scaler = StandardScaler().fit(x_train)

rescaledx = scaler.transform(x_train)

model = SVR(C=2.0)

model.fit(rescaledx, y_train)



rescaledvalidationx = scaler.transform(x_validation)

predictions = model.predict(rescaledvalidationx)

print(mean_squared_error(y_validation, predictions))