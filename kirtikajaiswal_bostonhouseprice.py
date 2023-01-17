import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas import read_csv

from pandas import set_option

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
# Load Data

filename = "/kaggle/input/boston-house-prices/housing.csv"

colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv(filename, names=colnames, delim_whitespace=True)
# Upper 20 observations

data.head(20)
# shape of data

data.shape
# data types of attributes

data.dtypes
# statistical properties of attributes

set_option('precision', 3)

data.describe()
# Correlations between attributes

correlations = data.corr(method='pearson')

correlations
# Split inputs and labels

data_split = data.values

inputs = data_split[:, 0:13]

labels = data_split[:, 13]
# Scale inputs in range of 0 to 1

scaler = MinMaxScaler(feature_range=(0,1))

rescaled_inputs = scaler.fit_transform(inputs)
# Scale inputs to mean = 0 to std = 1

scaler = StandardScaler().fit(rescaled_inputs)

scaled_inputs = scaler.transform(rescaled_inputs)
# Split data for testing and training

input_train, input_test, label_train, label_test = train_test_split(scaled_inputs, labels, test_size=0.2, random_state=7, shuffle=True)
# Compare various regression models to finalize the best one

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

num_folds = 10

seed = 7

scoring = 'neg_mean_squared_error'

models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    cv_results = cross_val_score(model, input_train, label_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare various ensemble methods to finalize the best one

ensembles = []

ensembles.append(('AB', AdaBoostRegressor()))

ensembles.append(('GBM',GradientBoostingRegressor()))

ensembles.append(('RF', RandomForestRegressor()))

ensembles.append(('ET', ExtraTreesRegressor()))

results = []

names = []

for name, model in ensembles:

    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    cv_results = cross_val_score(model, input_train, label_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Best number of estimators 

param_grid = dict(n_estimators=np.arange(50,400,50))

model = ExtraTreesRegressor(random_state=seed)

kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(input_train, label_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Train model

model = ExtraTreesRegressor(random_state=seed, n_estimators=150)

model.fit(input_train, label_train)
# Test model on testing data

prediction = model.predict(input_test)

print(mean_squared_error(label_test, prediction))