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
import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from sklearn import metrics

import gc



pd.set_option('display.max_columns', 200)
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
train_df.target.value_counts()
param = {

        'num_leaves': 10,

        'max_bin': 119,

        'min_data_in_leaf': 11,

        'learning_rate': 0.02,

        'min_sum_hessian_in_leaf': 0.00245,

        'bagging_fraction': 1.0, 

        'bagging_freq': 5, 

        'feature_fraction': 0.05,

        'lambda_l1': 4.972,

        'lambda_l2': 2.276,

        'min_gain_to_split': 0.65,

        'max_depth': 14,

        'save_binary': True,

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

    }
nfold = 5
target = 'target'

predictors = train_df.columns.values.tolist()[2:]
gc.collect()
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)



oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))



i = 1

for train_index, valid_index in skf.split(train_df, train_df.target.values):

    print("\nfold {}".format(i))

    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,

                           label=train_df.iloc[train_index][target].values,

                           feature_name=predictors,

                           free_raw_data = False

                           )

    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,

                           label=train_df.iloc[valid_index][target].values,

                           feature_name=predictors,

                           free_raw_data = False

                           )   



    

    clf = lgb.train(param, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=50, early_stopping_rounds = 50)

    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 

    

    predictions += clf.predict(test_df[predictors], num_iteration=clf.best_iteration) / nfold

    i = i + 1



print("\n\nCV AUC: {:<0.2f}".format(metrics.roc_auc_score(train_df.target.values, oof)))
sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub_df["target"] = predictions

sub_df[:10]
sub_df.to_csv("Customer_Transaction.csv", index=False)