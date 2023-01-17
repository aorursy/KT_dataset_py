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
dataset_train.head()
X_train = dataset_train.iloc[:, dataset_train.columns != 'target'].values

y_train = dataset_train.iloc[:, 1].values

X_test = dataset_test.values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X_train[:,0] = le.fit_transform(X_train[:,0])

X_test[:,0] = le.fit_transform(X_test[:,0])
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
pd.concat([dataset_test.ID_code, pd.Series(y_pred_gnb).rename('target')], axis = 1).to_csv('gnb_submission.csv')
dataset_gnb = pd.concat([dataset_test.ID_code, pd.Series(y_pred_gnb).rename('target')], axis = 1)
dataset_gnb.target.value_counts()
from sklearn import tree
tree = tree.DecisionTreeClassifier()

tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
dataset_tree = pd.concat((dataset_test.ID_code, pd.Series(y_pred_tree).rename('target')), axis = 1)

dataset_tree
dataset_tree.to_csv('tree_submission_final.csv')
sample_sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sample_sub
#pandas

#numpy

#the train/test split

# Import xgboost

import xgboost as xgb

import pandas as pd
xg_cl = xgb.XGBClassifier(objective = 'reg:logistic', n_estimators = 400, seed=132 )
xg_cl.fit(X_train, y_train)
y_pred_xg = xg_cl.predict(X_test)
dataset_xg = pd.concat((dataset_test.ID_code, pd.Series(y_pred_xg).rename('target')), axis = 1)

dataset_xg
dataset_xg.to_csv('xgboost_submission.csv')


from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold
n_estimators = range(50, 400, 50)

param_grid = dict(n_estimators=n_estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

model = xgb.XGBClassifier()
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)

grid_result = grid_search.fit(X_train, y_train)