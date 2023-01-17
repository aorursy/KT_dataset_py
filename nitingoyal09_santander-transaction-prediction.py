import time

import random

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Basic

import datetime

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV

from sklearn import metrics

import statsmodels as sm

from sklearn.pipeline import make_pipeline



# Model considerations

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

from catboost import CatBoostClassifier, Pool

import xgboost as xgb

import eli5

from eli5.sklearn import PermutationImportance



# Resampling 

from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.svm import LinearSVC



# Input

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train.head()
x = train.drop(['ID_code', 'target'], axis = 1)

y = train['target']
x_test = test.drop(['ID_code'], axis = 1)
x.head()
# how much training data we have

x.shape
train.target.unique()
# % of each class, as target variable is binary

y0 = len(train[train.target == 0]) / len(train.target) * 100

y1 = len(train[train.target == 1]) / len(train.target) * 100

class_percentage = [y0, y1]
print("Percentage of Class0: %s \nPercentage of Class1: %s: " % (y0, y1))
label = ['y0', "y1"]

sns.barplot(label, class_percentage)
# let's check the variable distribution

x.describe()
# random column selection

# print('=====Randomly selected Column Distributions=====')

# plt.figure(figsize=(26, 24))

# columns = x.columns.values

# random.seed(32)

# col_random = random.SystemRandom()

# for i in range(0, 32):

#     col = col_random.choice(columns)

#     plt.subplot(8, 4, i + 1)

#     sns.distplot(train[col], color = 'seagreen')

#     plt.axvline(train[col].mean(), 0, 1, linestyle = '--', color = 'blue')

#     plt.title(col)
# random column selection for outliers check using BOXPlot

# plt.figure(figsize=(20, 40))

# for i in range(0, 32):

#     col = col_random.choice(columns)

#     plt.subplot(16, 2, i+1)

#     sns.boxplot(x[col], color='orange')

#     plt.title(col)
# correlation graph of randomly selected two variables

# plt.figure(figsize = (24, 24))

# for i in range(0, 12):

#     col1 = col_random.choice(columns)

#     col2 = col_random.choice(columns)

#     plt.subplot(6, 4, i+1)

#     sns.regplot(x=x[col1][0:2000], y=x[col2][0:2000], data=x);

#     plt.title(col1+ ' v/s ' +col2)
# params = {

#         'num_leaves': 6,

#         'max_bin': 63,

#         'min_data_in_leaf': 17,

#         'learning_rate': 0.019,

#         'min_sum_hessian_in_leaf': 0.000446,

#         'bagging_fraction': 0.81, 

#         'bagging_freq': 5, 

#         'lambda_l1': 4.218,

#         'lambda_l2': 1.734,

#         'min_gain_to_split': 0.1501,

#         'max_depth': 14,

#         'save_binary': True,

#         'seed': 42,

#         'feature_fraction_seed': 42,

#          'feature_fraction': 0.85,

#         'bagging_seed': 42,

#         'drop_seed': 42,

#         'data_random_seed': 42,

#         'objective': 'binary',

#         'boosting_type': 'gbdt',

#         'verbose': 1,

#         'metric': 'auc',

#         'is_unbalance': True,

#         'boost_from_average': False,

#     }



# n_fold = 5

# folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
# for fold_n, (train_index, valid_index) in enumerate(folds.split(x,y)):

#     print('Fold', fold_n, 'started at', time.ctime())

    

#     X_train, X_valid = x.iloc[train_index], x.iloc[valid_index]

#     y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

#     train_data = lgb.Dataset(X_train, label=y_train)

#     valid_data = lgb.Dataset(X_valid, label=y_valid)

    

#     model = lgb.train(params,train_data,

#                       num_boost_round = 20000,

#                       valid_sets = [train_data, valid_data],

#                       verbose_eval = 300,

#                       early_stopping_rounds = 200)
# predictions = model.predict(x_test)

# sub = pd.read_csv('../input/sample_submission.csv')

# sub['target'] = predictions

# sub.to_csv('lgb.csv', index=False)
# # model - 2 Catboost

# train_pool = Pool(X_train, y_train)

# m = CatBoostClassifier(iterations=300, eval_metric="AUC", boosting_type = 'Ordered', task_type = "GPU")

# m.fit(X_train, y_train, silent=True)

# score = m.score(X_valid, y_valid)

# print(score)

# predictions1 = m.predict(x_test)
# sub['target'] = predictions1

# sub.to_csv('catboost.csv', index=False)
# model - 3 Random Foreset - LB score - 0.506 (Not Satisfactory!)

# X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, stratify=y)

# model = RandomForestClassifier()

# cv_scores = cross_val_score(model,X_train,y_train,scoring='accuracy')

# model.fit(X_train, y_train)

# score = model.score(X_valid, y_valid)

# print(score)

# predictions2 = model.predict(x_test)
# sub['target'] = predictions2

# sub.to_csv('randomforest.csv', index=False)
# Resampling the data

sampler = SMOTE(random_state = 0)

X_resampled, y_resampled = SMOTE().fit_resample(x, y)
# Model - 4: lgb with resampled data

# for fold_n, (train_index, valid_index) in enumerate(folds.split(X_resampled,y_resampled)):

#     print('Fold', fold_n, 'started at', time.ctime())

#     X_train, X_valid = x.iloc[train_index], x.iloc[valid_index]

#     y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

#     train_data = lgb.Dataset(X_train, label=y_train)

#     valid_data = lgb.Dataset(X_valid, label=y_valid)

    

#     model = lgb.train(params,train_data,

#                       num_boost_round = 20000,

#                       valid_sets = [train_data, valid_data],

#                       verbose_eval = 300,

#                       early_stopping_rounds = 200)
# predictions3 = model.predict(x_test)

# sub['target'] = predictions3

# sub.to_csv('lgb1.csv', index=False)
# # Model - 5 CatBoost with resampled data

# X_train_re, X_valid_re, y_train_re, y_valid_re = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled)

# train_pool = Pool(X_train_re, y_train_re)

# m1 = CatBoostClassifier(iterations=300, eval_metric="AUC", boosting_type = 'Ordered', task_type = "GPU")

# m1.fit(X_train_re, y_train_re, silent=True)

# score = m1.score(X_valid_re, y_valid_re)

# print(score)
# # One more test could be done with data by resampling the training data and validating the model on the fresh data.

# predictions4 = m1.predict(x_test)

# sub['target'] = predictions4

# sub.to_csv('catboost1.csv', index=False)
# Model - 6 RandomForest With resampled data - score improves to 0.527 (Better but still not satisfactory)

# m2 = RandomForestClassifier()

# cv_scores = cross_val_score(m2,X_train_re,y_train_re,scoring='accuracy')

# m2.fit(X_train_re, y_train_re)

# score = m2.score(X_valid_re, y_valid_re)

# print(score)
# predictions5 = m2.predict(x_test)

# sub['target'] = predictions5

# sub.to_csv('randomforest1.csv', index=False)
# Outliers Handling

# Data Standardization using Z-score Normalization

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
# explore the feature importance

xg_cls = xgb.XGBClassifier(max_depth=50, 

                          min_child_weight=1,  

                          n_estimators=200,

                          n_jobs=-1 , 

                          verbose=1,

                          learning_rate=0.16)

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, stratify=y)

xg_cls.fit(X_train,y_train)
print(xg_cls.score(X_valid, y_valid))
predictions5 = xg_cls.predict(x_test)

sub['target'] = predictions5

sub.to_csv('xg_cls.csv', index=False)