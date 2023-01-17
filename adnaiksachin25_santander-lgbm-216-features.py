# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from sklearn import metrics

import gc



pd.set_option('display.max_columns', 200)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Lets import data sets

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
print(train_df.head())

print(test_df.head())
predictors=train_df.columns[2:]

predictors
pos=predictors[train_df[predictors].mean()>0]

neg=predictors[train_df[predictors].mean()<=0]
idx = features = pos

for df in [train_df, test_df]:

    df['sum_pos'] = df[idx].sum(axis=1)  

    df['min_pos'] = df[idx].min(axis=1)

    df['max_pos'] = df[idx].max(axis=1)

    df['mean_pos'] = df[idx].mean(axis=1)

    df['std_pos'] = df[idx].std(axis=1)

    df['skew_pos'] = df[idx].skew(axis=1)

    df['kurt_pos'] = df[idx].kurtosis(axis=1)

    df['med_pos'] = df[idx].median(axis=1)
idx = features = neg

for df in [train_df, test_df]:

    df['sum_neg'] = df[idx].sum(axis=1)  

    df['min_neg'] = df[idx].min(axis=1)

    df['max_neg'] = df[idx].max(axis=1)

    df['mean_neg'] = df[idx].mean(axis=1)

    df['std_neg'] = df[idx].std(axis=1)

    df['skew_neg'] = df[idx].skew(axis=1)

    df['kurt_neg'] = df[idx].kurtosis(axis=1)

    df['med_neg'] = df[idx].median(axis=1)
train_df.head()
test_df.head()
param = {

    'num_leaves': 25,

     'max_bin': 60,

     'min_data_in_leaf': 5,

     'learning_rate': 0.010614430970330217,

     'min_sum_hessian_in_leaf': 0.0093586657313989123,

     'feature_fraction': 0.056701788569420042,

     'lambda_l1': 0.060222413158420585,

     'lambda_l2': 4.6580550589317573,

     'min_gain_to_split': 0.29588543202055562,

     'max_depth': 50,

     'save_binary': True,

     'seed': 1234,

     'feature_fraction_seed': 1234,

     'bagging_seed': 1234,

     'drop_seed': 1234,

     'data_random_seed': 1234,

     'objective': 'binary',

     'boosting_type': 'gbdt',

     'verbose': 1,

     'metric': 'auc',

     'is_unbalance': True,

     'boost_from_average': False

}
# number of folds

nfold = 10 
target = 'target'

predictors = train_df.columns.values.tolist()[2:]
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



    nround = 8000

    clf = lgb.train(param, xg_train, nround, valid_sets = [xg_valid], verbose_eval=250)

    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=nround) 

    

    predictions += clf.predict(test_df[predictors], num_iteration=nround) / nfold

    i = i + 1



print("\n\nCV AUC: {:<0.4f}".format(metrics.roc_auc_score(train_df.target.values, oof)))
submission = pd.DataFrame({"ID_code": test_df.ID_code.values})

submission["target"] = predictions

submission[:10]
submission.to_csv("LGMB_210_featutes.csv", index=False)