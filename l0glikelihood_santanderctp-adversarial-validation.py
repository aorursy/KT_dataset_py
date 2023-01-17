import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import time

import datetime

from bayes_opt import BayesianOptimization

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

import gc

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train['target'] = 0

test['target'] = 1

new_dataset = train.append(test).sample(frac=1,random_state=11).reset_index(drop=True)
target = new_dataset['target'].values

del new_dataset['target']

del new_dataset['ID_code']
train_adv, val_adv, train_adv_target, val_adv_target = train_test_split(new_dataset, target, 

                                                               test_size=0.2, random_state=11)
params = {'num_leaves': 100,

          'objective': 'binary',

          'learning_rate': 0.05,

          'boosting':'gbdt',

          'feature_fraction': 0.155,

          'bagging_fraction': 0.8434,

          'bagging_seed':11,

          'max_depth': -1,

          'lambda_l1': 3.858,

          'lambda_l2': 1.181,

          'min_split_gain': 0.04322,

          'min_child_weight': 0.08105,

          'random_state': 11,

          'metric': 'auc',

          'verbosity': -1}



train_adv = lgb.Dataset(train_adv, label=train_adv_target)

val_adv = lgb.Dataset(val_adv, label=val_adv_target)

num_round = 2000

clf = lgb.train(params, train_adv, num_round, valid_sets = [train_adv, val_adv], verbose_eval=50, early_stopping_rounds = 200)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),new_dataset.columns), reverse=True), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-01.png')