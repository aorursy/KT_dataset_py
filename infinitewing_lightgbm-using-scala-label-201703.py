import lightgbm as lgb

import pandas as pd

import numpy as np

from sklearn import *



# scala label

train = pd.read_csv('../input/kkbox-churn-scala-label/user_label_201703.csv', dtype={'is_churn': 'int8'})



# train_v2

#train = pd.read_csv('../input/kkbox-churn-prediction-challenge/train_v2.csv')
test = pd.read_csv('../input/kkbox-churn-prediction-challenge/sample_submission_v2.csv')



members = pd.read_csv('../input/kkbox-churn-prediction-challenge/members_v3.csv')

train = pd.merge(train, members, how='left', on='msno')

test = pd.merge(test, members, how='left', on='msno') 

gender = {'male':1, 'female':2}

train['gender'] = train['gender'].map(gender)

test['gender'] = test['gender'].map(gender)

train = train.fillna(-1)

test = test.fillna(-1)



cols = [c for c in train.columns if c not in ['is_churn','msno']]

print(cols)
lgb_params = {

    'learning_rate': 0.05,

    'application': 'binary',

    'max_depth': 5,

    'num_leaves': 128,

    'verbosity': -1,

    'metric': 'binary_logloss'

}

x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.2, random_state=0)

        

# lgb

d_train = lgb.Dataset(x1, label=y1)

d_valid = lgb.Dataset(x2, label=y2)

watchlist = [d_train, d_valid]



model = lgb.train(lgb_params, train_set=d_train, num_boost_round=240, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=10) 

lgb_pred = model.predict(test[cols])



test['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15)

test[['msno','is_churn']].to_csv('lgb_sub_scala.csv', index=False)