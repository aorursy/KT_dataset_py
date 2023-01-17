import numpy as np

import pandas as pd

import os

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

import lightgbm as lgb

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')



random_state = 0

np.random.seed(random_state)

df_train = pd.read_csv('../input/train.csv')



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

    "bagging_seed" : random_state,

    "verbosity" : 1,

    "seed": random_state

}



train_target = df_train['target']



col_list = []

val_auc_list = []

for i in ['var_{}'.format(c) for c in range(0,200)]:

    print(i)

    train_features = df_train.loc[:,i]

    x_train,x_valid,y_train,y_valid = train_test_split(train_features,train_target, test_size=0.2, stratify=train_target, random_state = random_state)



    trn_data = lgb.Dataset(x_train.values.reshape(-1,1), label=y_train)

    val_data = lgb.Dataset(x_valid.values.reshape(-1,1), label=y_valid)

    lgb_clf = lgb.train(lgb_params,

                        trn_data,100000,

                        valid_sets = [trn_data, val_data],

                        early_stopping_rounds=3000,

                        verbose_eval = 1500)

    col_list.append(i)

    val_auc_list.append(roc_auc_score(y_valid.values,lgb_clf.predict(x_valid.values.reshape(-1,1))))

    

df = pd.DataFrame({'col' : col_list, 'auc' : val_auc_list}, columns=['col', 'auc'])



df.to_csv('single_pred.csv', index = False)

print(df.sort_index(by = ['auc'], ascending=False))