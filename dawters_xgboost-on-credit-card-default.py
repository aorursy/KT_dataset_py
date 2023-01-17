# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as skl

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
data.head()
data.describe()
from sklearn.model_selection import train_test_split 



X, y = data.drop(['default.payment.next.month'], axis=1), data['default.payment.next.month']



test_prop = 0.2



X_train, X_test, y_train, y_test = train_test_split(

     X, y, test_size=test_prop)





D_train = xgb.DMatrix(X_train, label=y_train)

D_test = xgb.DMatrix(X_test, label=y_test)
import xgboost as xgb



params = {

    # Parameters that we are going to tune.

    'max_depth':6,

    'min_child_weight': 1,

    'eta':.3,

    'subsample': 1,

    'colsample_bytree': 1,

    # Other parameters

    'objective':'binary:hinge',

}



steps = 50 # The number of training iterations



model = xgb.train(params, D_train, steps)
preds = model.predict(D_test)
from sklearn.metrics import roc_auc_score



auc = roc_auc_score(preds, y_test)



print('ROC AUC:', auc)
"""

from sklearn.model_selection import GridSearchCV



estimator = xgb.XGBClassifier(

    objective= 'binary:hinge'

)



var_params = {

    # Parameters that we are going to tune.

    'max_depth':range(6,10,2),

    'min_child_weight': range(3,9,2),

    'eta':[0.005,0.01,0.05,0.1,0.2],

    'colsample_bytree': [0.6,0.8,1],

    'subsample':[0.8,1]

}



grid_search = GridSearchCV(

    estimator=estimator,

    param_grid=var_params,

    scoring = 'roc_auc',

    n_jobs = 3,

    cv = 10,

    verbose=True

)



grid_search.fit(X_train, y_train)"""
grid_search.best_estimator_
parameters0 = {

    # Parameters that we are going to tune.

    'max_depth':8,

    'min_child_weight': 3,

    'eta':.005,

    'subsample': 1,

    'colsample_bytree': 1,

    # Other parameters

    'objective':'binary:hinge',

}



model_opt = xgb.train(parameters0, D_train, steps)



preds0 = model.predict(D_test)
auc0 = roc_auc_score(preds0, y_test)



print('New ROC AUC:', auc0)