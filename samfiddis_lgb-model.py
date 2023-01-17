# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

features = train.columns.values[2:202]

x_train = train[features].values

y_train = train.target.values



skf = StratifiedKFold(n_splits=10)

skf.get_n_splits(x_train, y_train)
random_state = 2567

lgb_params = {

    "objective" : "binary",

    "metric" : "auc",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 13,

    "learning_rate" : 0.01,

    "bagging_freq": 5,

    "bagging_fraction" : 0.4,

    "feature_fraction" : 0.05,

    "min_data_in_leaf": 80,

    "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    #"lambda_l1" : 5,

    #"lambda_l2" : 5,

    "bagging_seed" : random_state,

    "verbosity" : 1,

    "seed": random_state

}

oof = np.zeros(len(train))

predictions = np.zeros(len(test))



for train_index, test_index in skf.split(x_train, y_train):

    print("TRAIN:", train_index, "TEST:", test_index)

    x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]

    y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

    

    trn_data = lgb.Dataset(x_train_cv, label=y_train_cv)

    val_data = lgb.Dataset(x_test_cv, label=y_test_cv)

    

    num_round = 10000

    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)

    oof[test_index] = clf.predict(x_test_cv, num_iteration=clf.best_iteration)

    

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / 10
sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv("lgb_submission.csv", index=False)