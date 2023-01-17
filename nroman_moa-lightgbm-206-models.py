import pandas as pd

import numpy as np

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import log_loss

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline
files = ['../input/lish-moa/test_features.csv', 

         '../input/lish-moa/train_targets_scored.csv',

         '../input/lish-moa/train_features.csv',

         '../input/lish-moa/train_targets_nonscored.csv',

         '../input/lish-moa/sample_submission.csv']



with multiprocessing.Pool() as pool:

    test, train_target, train, train_nonscored, sub = pool.map(pd.read_csv, files)
# One-Hot encoding

for feature in ['cp_time', 'cp_type', 'cp_dose']:

    concat = pd.concat([train[feature], test[feature]], ignore_index=True)

    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix=feature)

    train = pd.concat([train, dummies.iloc[:train.shape[0]]], axis=1)

    test = pd.concat([test, dummies.iloc[:test.shape[0]]], axis=1)
targets = [col for col in train_target.columns if col != 'sig_id']

print('Number of different labels:', len(targets))
features = [col for col in train.columns if col not in ['sig_id', 'cp_time', 'cp_type', 'cp_dose']]

print('Number of features:', len(features))
X = train[features]
params = {'num_leaves': 491,

          'min_child_weight': 0.03,

          'feature_fraction': 0.3,

          'bagging_fraction': 0.4,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'binary_logloss',

          "verbosity": 0,

          'reg_alpha': 0.4,

          'reg_lambda': 0.6,

          'random_state': 47

         }
accumulative_loss = 0

skf = StratifiedKFold(n_splits=3, random_state=47, shuffle=True)



print('Execution time | Model number | logloss | new logloss | best coeff')

# 206 different models. One for each label

for model, target in enumerate(targets, 1):

    y = train_target[target]

    start_time = time()

    preds = np.zeros(test.shape[0])

    oof = np.zeros(X.shape[0])



    for trn_idx, test_idx in skf.split(X, y):

        

        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

        val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])

        clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds=20)

        oof[test_idx] = clf.predict(X.iloc[test_idx])

        preds += clf.predict(test[features]) / skf.n_splits



    loss = log_loss(y, oof)

    

    # Hacking the metric

    coeffs = [3, 2, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7]

    best_coeff = 0

    best_loss = loss

    for coeff in coeffs:

        new_oof = oof.copy()

        new_oof[new_oof < new_oof.mean() / coeff] = 0

        new_loss = log_loss(y, new_oof)

        if new_loss < loss:

            preds[preds < preds.mean() / coeff] = 0

            best_coeff = coeff

            best_loss = new_loss

    

    if best_coeff:

        preds[preds < preds.mean() / best_coeff] = 0

    # End of metric hacking

    sub[target] = preds



    accumulative_loss += best_loss

    print('{}\t\t{}\t{:.5f}\t\t{:.5f}\t\t{}'.format(str(datetime.timedelta(seconds=time() - start_time))[:7], model, loss, best_loss, best_coeff))

    del preds, oof, start_time, y, loss, best_loss, new_oof

    gc.collect();
print('Overall mean loss: {:.5f}'.format(accumulative_loss / 206))
sub.to_csv('submission.csv', index=False)