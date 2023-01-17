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

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.feature_selection import RFE

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
# Load data

train = pd.read_csv("../input/train.csv").drop('id', axis=1)

test = pd.read_csv('../input/test.csv')
# Manage data on train set

y_train = train['target']

X_train = train.drop('target', axis=1)



# Manage data on test set

X_test = test.drop('id', axis = 1)
# Normalize data

std = StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.fit_transform(X_test)
# GridSearchCV with XGBoost model

xgb_model = xgb.XGBClassifier(random_state=42)

param_grid = {'objective':['binary:logistic'],

              'learning_rate': [0.001,0.05,0.1, 10], 

              'max_depth': [2,3,4,5,6],

              'min_child_weight': [11],

              'subsample': [0.8],

              'colsample_bytree': [0.7],

              'n_estimators': [1000]}



grid = GridSearchCV(estimator = xgb_model, cv=5, param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1, refit=True)

grid.fit(X_train,y_train)



print("Best Score:" + str(grid.best_score_))

print("Best Parameters: " + str(grid.best_params_))



best_parameters = grid.best_params_
# XGBoost model with RFE and 70 features

xgb_model = xgb.XGBClassifier(**best_parameters)

xgb_model.fit(X_train,y_train)



selector = RFE(xgb_model, 200, step=1)

selector.fit(X_train,y_train)



xgb_preds = selector.predict_proba(X_test)[:,1]



train_predict = selector.predict(X_train)

roc_auc_score(y_train, train_predict)
# GridSearchCV with LogisticRegressor model

lr = LogisticRegression(random_state=42)

param_grid = {'class_weight' : ['balanced', None], 

              'penalty' : ['l2','l1'], 

              'C' : [0.001, 0.01, 0.1, 1, 10, 100],

              'solver': ['saga']}



grid = GridSearchCV(estimator = lr, cv=5, param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)

grid.fit(X_train,y_train)



print("Best Score:" + str(grid.best_score_))

print("Best Parameters: " + str(grid.best_params_))



best_parameters = grid.best_params_
# LogisticRegression with RFE and 150 features

lr = LogisticRegression(**best_parameters)

lr.fit(X_train,y_train)



selector = RFE(lr, 200, step=1)

selector.fit(X_train,y_train)



lr_preds = lr.predict_proba(X_test)[:,1]



train_predict = lr.predict(X_train)

roc_auc_score(y_train, train_predict)
# Combine models

final_preds = (lr_preds * 0.8 + xgb_preds * 0.2)

submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = final_preds

submission.to_csv('submission.csv', index=False)

submission.head(20)