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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

dataset_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

dataset_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
X_train = dataset_train.iloc[:, dataset_train.columns != 'target']

y_train = dataset_train.iloc[:, 1].values

X_test = dataset_test.iloc[:, dataset_test.columns != 'ID_code'].values
X_train = X_train = X_train.iloc[:, X_train.columns != 'ID_code'].values
from sklearn.linear_model import LogisticRegression
log_reg_cls = LogisticRegression()
log_reg_cls.fit(X_train, y_train)
y_preds_log_reg = log_reg_cls.predict(X_test)
dataset_log_reg = pd.concat((dataset_test.ID_code, pd.Series(y_preds_log_reg).rename('target')), axis = 1)

dataset_log_reg.target.value_counts()
dataset_log_reg.to_csv('log_reg_better_submission.csv', index=False)
# Import xgboost

import xgboost as xgb

xg_cl = xgb.XGBClassifier(

                          objective = 'binary:logistic', 

                          n_estimators = 10000, seed=123, 

                          learning_rate=0.25,

                          max_depth=2,

                          colsample_bytree=0.35,

                          subsample=0.82,

                          min_child_weight= 53,

                          gamma=9.9,

                         )
xg_cl.fit(X_train, y_train)
y_pred_xg = xg_cl.predict(X_test)
dataset_xg = pd.concat((dataset_test.ID_code, pd.Series(y_pred_xg).rename('target')), axis = 1)

dataset_xg.target.value_counts()
dataset_xg.to_csv('xg_boost4_upd_submission.csv', index=False)