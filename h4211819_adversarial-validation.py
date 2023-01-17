# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')

from tqdm import tqdm

from sklearn.preprocessing import scale, minmax_scale

from scipy.stats import norm
random_state = 42

np.random.seed(random_state)

train = pd.read_csv('../input/train.csv')[:]

test = pd.read_csv('../input/test.csv')[:]



features = [c for c in train.columns if c not in ['id', 'target']]



len_train = len(train)

train['target'] = 1

train = train.append(test).reset_index(drop = True)

train['target'] = train['target'].fillna(0)
lgb_params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'verbose': 1,

    'learning_rate': 0.05,

    'num_leaves': 31,

    'feature_fraction': 0.7,

    'min_data_in_leaf': 200,

    'bagging_fraction': 0.8,

    'bagging_freq': 20,

    'min_hessian': 0.01,

    'feature_fraction_seed': 2,

    'bagging_seed': 3,

    "seed": random_state

}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

oof = train[['id', 'target']]

oof['predict'] = 0

val_aucs = []
for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):

    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']

    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

    trn_data = lgb.Dataset(X_train, label=y_train)

    val_data = lgb.Dataset(X_valid, label=y_valid)

    evals_result = {}

    lgb_clf = lgb.train(lgb_params,

                        trn_data,

                        7500,

                        valid_sets=[val_data],

                        early_stopping_rounds=100,

                        verbose_eval=50,

                        evals_result=evals_result)



    p_valid = lgb_clf.predict(X_valid[features], num_iteration=lgb_clf.best_iteration)



    oof['predict'][val_idx] = p_valid

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

mean_auc = np.mean(val_aucs)

std_auc = np.std(val_aucs)

all_auc = roc_auc_score(oof['target'], oof['predict'])

print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))