import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

pd.set_option('display.max_columns', 1000)

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import lightgbm as lgb

import gc

import datetime

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sns.set()

%matplotlib inline
sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

print('Successfully loaded sample_submisssion')
import pickle



with open('/kaggle/input/ieee-fraud-detection/train.pickle', 'rb') as f1, open('/kaggle/input/ieee-fraud-detection/output/test.pickle', 'rb') as f2:

    train = pickle.load(f1)

    test = pickle.load(f2)
# train = train.fillna(-999)

# test = test.fillna(-999)

# train = train.replace(np.inf,999)

# test = test.replace(np.inf,999)
X = train.drop(['isFraud'], axis = 1)

y = train['isFraud']



print('Our train set have {} columns'.format(train.shape[1]))

print('Our test set have {} columns'.format(test.shape[1]))



gc.collect()



y.value_counts()
params = {

                    'objective':'binary',

                    'boosting_type':'gbdt',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.005,

                    'num_leaves': 2**8,

                    'max_depth':-1,

                    'tree_learner':'serial',

                    'colsample_bytree': 0.7,

                    'subsample_freq':1,

                    'subsample':0.7,

                    'n_estimators':100000,

                    'max_bin':255,

                    'verbose':-1,

                    'random_state': 47,

                    'early_stopping_rounds':100,

                    'scale_pos_weight': 27.58,

                }
NFOLDS = 10

folds = KFold(n_splits=NFOLDS)





splits = folds.split(X, y)

y_preds = np.zeros(test.shape[0])

y_oof = np.zeros(X.shape[0])

score = 0



for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label = y_train)

    dvalid = lgb.Dataset(X_valid, label = y_valid)

    

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], 

                    verbose_eval = 200, early_stopping_rounds = 500)

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clf.predict(test) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print("Mean AUC: ", score)

print("Out of folds AUC: ", roc_auc_score(y, y_oof))
sub['isFraud'] = y_preds

sub.to_csv('submission_v1.csv', index = False)