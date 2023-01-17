# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
%%time
files = ['../input/lish-moa/test_features.csv', 
         '../input/lish-moa/train_targets_scored.csv',
         '../input/lish-moa/train_features.csv',
         '../input/lish-moa/train_targets_nonscored.csv',
         '../input/lish-moa/sample_submission.csv']

def load_data(file):
    return pd.read_csv(file)

with multiprocessing.Pool() as pool:
    test, train_target, train, train_nonscored, sub = pool.map(load_data, files)
print('Number of records:',len(train_target))
targets = [col for col in train_target.columns if col != 'sig_id']
print('Number of unique labels:', len(targets))
features = [col for col in train.columns if col != 'sig_id']
print('Number of features:', len(features))
for feature in ['cp_type', 'cp_dose']:
    le = LabelEncoder()
    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
    train[feature] = le.transform(list(train[feature].astype(str).values))
    test[feature] = le.transform(list(test[feature].astype(str).values))

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

# 206 different models. One for each label
for model, target in enumerate(targets, 1):
    y = train_target[target]
    start_time = time()
    preds = np.zeros(test.shape[0])
    oof = np.zeros(X.shape[0])

    for trn_idx, test_idx in skf.split(X, y):
        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
        clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds=5)
        oof[test_idx] = clf.predict(X.iloc[test_idx])
        preds += clf.predict(test[features]) / skf.n_splits

    sub[target] = preds
    loss = log_loss(y, oof)
    accumulative_loss += loss
    print('[{}] Model: {} logloss: {:.3f}'.format(str(datetime.timedelta(seconds=time() - start_time))[:7], model, loss))

    del preds, oof, start_time, y, loss
    gc.collect();
print('Overall mean loss: {:.3f}'.format(accumulative_loss / 206))
sub.to_csv('submission.csv', index=False)
