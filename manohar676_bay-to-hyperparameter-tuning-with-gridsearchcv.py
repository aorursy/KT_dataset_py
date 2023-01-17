from pprint import pprint

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

import numpy as np

import pandas as pd

import os

import sys

from IPython.display import Image
from sklearn.datasets import make_regression



X, y = make_regression(n_samples=500, n_features=4, n_informative=2,random_state=0, shuffle=False)



f,ax=plt.subplots(2,2,figsize=(14,14))



sns.scatterplot(x=X[:,0], y=y, ax=ax[0,0])

ax[0,0].set_xlabel('Feature 1 Values')

ax[0,0].set_ylabel('Y Values')

ax[0,0].set_title('Sactter Plot : Feature 1 vs Y')



sns.scatterplot(x=X[:,1], y=y,ax=ax[0,1])

ax[0,1].set_xlabel('Feature 2 Values')

ax[0,1].set_ylabel('Y Values')

ax[0,1].set_title('Sactter Plot : Feature 2 vs Y')



sns.scatterplot(x=X[:,2], y=y,ax=ax[1,0])

ax[1,0].set_xlabel('Feature 3 Values')

ax[1,0].set_ylabel('Y Values')

ax[1,0].set_title('Sactter Plot : Feature 3 vs Y')



sns.scatterplot(x=X[:,3], y=y,ax=ax[1,1])

ax[1,1].set_xlabel('Feature 4 Values')

ax[1,1].set_ylabel('Y Values')

ax[1,1].set_title('Sactter Plot : Feature 4 vs Y')



plt.show()
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer, mean_squared_error



rfr = RandomForestRegressor(verbose=0)

print('Parameters currently in use:\n')

pprint(rfr.get_params())

print('CV score with default parameters : ',-cross_val_score(rfr, X, y, cv=4, scoring = make_scorer(mean_squared_error, greater_is_better=False)).mean())
from sklearn.model_selection import GridSearchCV



# define the search space.

param_grid = {

    'bootstrap': [True],

    'max_depth': [50, 75, 100],

    'max_features': ['auto'],

    'min_samples_leaf': [1],

    'min_samples_split': [2],

    'n_estimators': [100,200,500,1000]}



# make scorer

MSE = make_scorer(mean_squared_error, greater_is_better=False)



# Configure the GridSearch model

model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1, cv=4, scoring=MSE, verbose=2)

# n_jobs=-1 : Means configured to use parallelism. use n_jobs=1 if use wish not to.



# Training

model.fit(X, y)



print('Random forest regression...')

print('Best Params:', model.best_params_)

print('Best CV Score:', -model.best_score_)
import xgboost as xgb



xgbr = xgb.XGBRegressor(seed=0)

# A parameter grid for XGBoost

param_grid = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }





model = GridSearchCV(estimator=xgbr, param_grid=param_grid, n_jobs=-1, cv=4, scoring=MSE)

model.fit(X, y)

score = -model.best_score_



print('eXtreme Gradient Boosting regression...')

print(xgbr)

print('Best Params:\n', model.best_params_)

print('Best CV Score:', score)