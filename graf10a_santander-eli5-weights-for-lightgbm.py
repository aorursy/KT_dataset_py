import os

import shutil

import feather

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance



NROUNDS = 1000000

EARLY_STOPPING = 3000

RANDOM_STATE = 44000



###############################################################

# Loading data

###############################################################



print("Loading and spliting data...")



train = feather.read_dataframe('../input/data-serialization/train.feather')

test = feather.read_dataframe('../input/data-serialization/test.feather')

y = feather.read_dataframe('../input/data-serialization/target.feather')

"""

trn_x, val_x, trn_y, val_y = train_test_split(train.values, y.values.ravel(), 

                                              random_state=RANDOM_STATE)

"""

trn_x, val_x, trn_y, val_y = train_test_split(train, y.values.ravel(), random_state=RANDOM_STATE)
print("Training the classifier...")



clf = lgb.LGBMClassifier(n_estimators=NROUNDS,

                         #**params, 

                         num_leaves=13,

                         boost_from_average=False,

                         objective='binary',

                         max_depth=-1,

                         learning_rate=0.01,

                         boosting='gbdt',

                         min_data_in_leaf=80,

                         bagging_freq=5,

                         bagging_fraction=0.4,

                         feature_fraction=0.05,

                         min_sum_hessian_in_leaf=10.0,

                         tree_learner='serial',

                         #bagging_seed=11,

                         #reg_alpha=5,

                         #reg_lambda=5,

                         metric='auc',

                         verbosity=1,

                         #subsample=0.81,

                         #min_gain_to_split=0.01077313523861969,

                         #min_child_weight=19.428902804238373,

                         num_threads=4)



     

clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=3000,

        early_stopping_rounds=EARLY_STOPPING)
print("Computing permtation importances...")



perm = PermutationImportance(clf, random_state=RANDOM_STATE).fit(val_x, val_y)
print("Computing the ELI5 weights for LightGBM...")



eli5.show_weights(perm, feature_names = val_x.columns.tolist(), top=200)