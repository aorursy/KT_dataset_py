# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import sys

import os

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import pydicom 
# Read the data

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
train.head()
test.head()
train['sex'].fillna('unknown', inplace=True)

test['sex'].fillna('unknown', inplace=True)



train['age_approx'].fillna(train['age_approx'].mode().values[0], inplace=True)

test['age_approx'].fillna(test['age_approx'].mode().values[0], inplace=True)



train['anatom_site_general_challenge'].fillna('unknown', inplace=True)

test['anatom_site_general_challenge'].fillna('unknown', inplace=True)
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()



train['sex_enc'] = enc.fit_transform(train.sex.astype('str'))

test['sex_enc'] = enc.transform(test.sex.astype('str'))



train['age_enc'] = enc.fit_transform(train.age_approx.astype('str'))

test['age_enc'] = enc.transform(test.age_approx.astype('str'))



train['anatom_enc'] = enc.fit_transform(train.anatom_site_general_challenge.astype('str'))

test['anatom_enc'] = enc.transform(test.anatom_site_general_challenge.astype('str'))
train.head()
train['age_enc'] = train['age_enc'] / np.mean(train['age_enc'])

test['age_enc'] = test['age_enc'] / np.mean(test['age_enc'])



train['anatom_enc'] = train['anatom_enc'] / np.mean(train['anatom_enc'])

test['anatom_enc'] = test['anatom_enc'] / np.mean(test['anatom_enc'])



train.head()
features = [

            'sex_enc',

            'age_enc',

            'anatom_enc'

]
X_train = train[features]

y_train = train['target']



x_test = test[features]
X_train.head()
x_test.head()
# model tuning



from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

import time

from sklearn.metrics import roc_auc_score



# A parameter grid for XGBoost

params = {

    'n_estimators':[500],

    'min_child_weight':[4,5], 

    'gamma':[i/10.0 for i in range(3,6)],  

    'subsample':[i/10.0 for i in range(6,11)],

    'colsample_bytree':[i/10.0 for i in range(6,11)], 

    'max_depth': [2,3,4,6,7],

    'objective': ['binary:logistic'],

    'booster': ['gbtree', 'gblinear'],

    'eval_metric': ['rmse'],

    'eta': [i/10.0 for i in range(3,6)],

}



reg = XGBRegressor(n_jobs=-1, nthread=-1)



# run randomized search

n_iter_search = 100

random_search = RandomizedSearchCV(reg, param_distributions=params,

                                   n_iter=n_iter_search, cv=5, iid=False, scoring='roc_auc')



start = time.time()

random_search.fit(X_train, y_train)

print("RandomizedSearchCV took %.2f seconds for %d candidates"

      " parameter settings." % ((time.time() - start), n_iter_search))
best_regressor = random_search.best_estimator_

preds = best_regressor.predict(x_test)
sample_submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

sample_submission.head()

sample_submission['target'] = preds

sample_submission.to_csv('submission.csv', index=False)