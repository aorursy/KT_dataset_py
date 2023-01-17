import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb # gradient boosting

import hyperopt as hp # optimization for Bayes

import sklearn # grid search and random search

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Look at our data

train.head(3)
test.head(3)
train = train.fillna(train.mean())
train.head(3)
train = pd.get_dummies(train) # Dummies without categorical vars
train.head(3)
#creating matrices for sklearn:

X_train = train[:train.shape[0]]

X_test = train[train.shape[0]:]

y = train.SalePrice
xgbr = xgb.XGBRegressor(n_estimators=100, n_jobs=-1) # our classification model
folds = 6

param_comb = 10
skf = sklearn.model_selection.StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
g_search = GridSearchCV(xgbr,

    {

        'max_depth': [3],

        'n_estimators': [6],

        'learning_rate': [0.2],

    }, n_jobs=4, cv=skf.split(X_train,y), verbose=3)



r_search = RandomizedSearchCV(xgbr,

    {

        'max_depth': [3],

        'n_estimators': [6],

        'learning_rate': [0.2],

    }, n_jobs=4, cv=skf.split(X_train,y), verbose=3)