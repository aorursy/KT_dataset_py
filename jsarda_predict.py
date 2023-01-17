#Librerias

import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

#Archivos

PATH="../input/"

train_df = pd.DataFrame.tail(pd.read_csv(PATH+"train.csv"),10000)

test_df = pd.read_csv(PATH+"test.csv")
#Formatos

train_df.shape, test_df.shape

#Parametros

param = {

        'num_leaves': 6,

        'max_bin': 63,

        'min_data_in_leaf': 45,

        'learning_rate': 0.01,

        'min_sum_hessian_in_leaf': 0.000446,

        'bagging_fraction': 0.55, 

        'bagging_freq': 5, 

        'max_depth': 14,

        'save_binary': True,

        'seed': 31452,

        'feature_fraction_seed': 31415,

         'feature_fraction': 0.51,

        'bagging_seed': 31415,

        'drop_seed': 31415,

        'data_random_seed': 31415,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

    }
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

target = train_df['target']
folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=31415)



oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))



feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 100



    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 250)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission.csv", index=False)