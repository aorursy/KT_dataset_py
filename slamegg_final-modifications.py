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
y_test = dataset_test.iloc[:, 1].values
X_train = X_train.iloc[:, X_train.columns != 'ID_code'].values
# Import xgboost
from xgboost import XGBClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
xgb = XGBClassifier()
# param_grid = {'max_depth': [3], 
#               'gamma': [0,9],
#               'n_estimators': [1000],
#               'tree_method': ['gpu_hist'],
#               'n_gpus': [1],
#               'colsample_bytree': [0.1,1],
#               'subsample': [0.82],
#               'scale_pos_weight': [8.951238929246692]
             
#              }
# with ignore_warnings(category=DeprecationWarning):
#     xgb_grid = GridSearchCV(xgb, param_grid, cv=10, refit=True, verbose=1, n_jobs=-1)
#     xgb_grid.fit(X_train,y_train)

# xgb_grid.best_estimator_
neg = len(y_train)-sum(y_train)
pos = sum(y_train)
scale_pos_weight  = float(neg/pos)
scale_pos_weight
XGB = XGBClassifier(scale_pos_weight=scale_pos_weight,
                        objective='binary:logistic',
                        random_state= 21,
                        subsample=0.83,
                        tree_method = 'gpu_hist',
                        learning_rate = 0.1, ## From initial gridsearch
                        n_estimators = 1000 ,  ## From initial gridsearch
                        tree_depth= 3     ## From initial gridsearch
                    )
XGB.fit(X_train, y_train)

y_pred_xg = XGB.predict(X_test)

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
y_test
accuracy_score(y_train, y_pred_xg)
dataset_xg = pd.concat((dataset_test.ID_code, pd.Series(y_pred_xg).rename('target')), axis = 1)
dataset_xg.target.value_counts()
dataset_xg.to_csv('xg_boost_gpu_newlast1_submission.csv', index=False)

from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()

GNB.fit(X_train,y_train)
y_preds_test = GNB.predict_proba(X_test)

probs_pos_test_gnb_one  = []
for pred in y_preds_test:
    probs_pos_test_gnb_one.append(pred[1])

log_reg_private = 0.85107
xgboost_private = 0.80526
gnb_private = 0.88763

log_reg_public = 0.84947
xgboost_public = 0.80901
gnb_public = 0.88848


# the histogram of the data
n, bins, patches = plt.hist(probs_pos_test_gnb_one, 50, density=1, facecolor='g', alpha=0.75)


plt.xlabel('Probability')
plt.ylabel('GNB_values')
plt.title('GNB')
plt.grid(True)
plt.show()
dataset_gnb = pd.concat((dataset_test.ID_code, pd.Series(probs_pos_test_gnb).rename('target')), axis = 1)
dataset_gnb.target.value_counts()
dataset_gnb.to_csv('gnb_submission.csv', index=False)

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
logist = LogisticRegression(C=0.001, class_weight='balanced')

logist.fit(X_train, y_train)
logist_pred = logist.predict_proba(X_test)
logist_pred
probs_pos_test_log  = []
for pred in logist_pred:
    probs_pos_test_log.append(pred[1])
 
# the histogram of the data
n, bins, patches = plt.hist(probs_pos_test_log, 50, density=1, facecolor='g', alpha=0.75)


plt.xlabel('Probability')
plt.ylabel('Log_reg_val')
plt.title('Log')
plt.grid(True)
plt.show()
dataset_log = pd.concat((dataset_test.ID_code, pd.Series(probs_pos_test_log).rename('target')), axis = 1)
dataset_log.target.value_counts()
dataset_log.to_csv('log_submission.csv', index=False)

names = ['log_reg', 'xg_boost', 'gnb']
values = [log_reg_private, xgboost_private, gnb_private]

plt.figure(figsize=(15, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
values

names = ['log_reg', 'xg_boost', 'gnb']
values = [log_reg_public, xgboost_public, gnb_public]

plt.figure(figsize=(15, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
values
