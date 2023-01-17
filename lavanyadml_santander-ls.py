import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import warnings

warnings.filterwarnings('ignore')# This Python 3 environment comes with many helpful analytics libraries installed



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
test_df.shape, train_df.shape
train_df.head()
train_df.isnull().values.any()
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

target = train_df['target']
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.33,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 12,

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1

}
num_round = 100000

# check random state 44000

folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=12345)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 2500)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
submission = pd.DataFrame({"ID_code": test_df.ID_code.values})

submission["target"] = predictions

submission.to_csv("submission.csv", index=False)
submission.head()