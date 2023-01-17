import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
train.head(10)
#すべて分析をかけると重いのでデータ数を絞る

max_row = 50000



# 0 ~ 50000行までのデータをtrain_pdpに詰める

train_pdp = train.iloc[:max_row, :]
import pandas_profiling as pdp

pdp.ProfileReport(train_pdp)
from sklearn.model_selection import train_test_split

from sklearn import metrics

import lightgbm as lgb
train_y = train['target']

train_x = train.drop(['ID_code','target'], axis=1)

test_x = test.drop('ID_code', axis=1)
train_X, eval_X, train_Y, eval_Y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)
lgb_train = lgb.Dataset(train_X, train_Y)

lgb_eval = lgb.Dataset(eval_X, eval_Y, reference=lgb_train)
params = {

    'objective': 'binary',

    'max_depth': 16,

    'learning_rate': 0.1,

    'is_unbalance': True,

    'random_state': 42,

    'metric': 'auc',

    'num_threads': 4}
gbm = lgb.train(params, 

                lgb_train,

                valid_sets=lgb_eval,

                num_boost_round=2000,

                verbose_eval=100,

                early_stopping_rounds = 100

               )
y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)   
submission_lgbm = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred

    })

submission_lgbm.to_csv('submission_lgbm.csv', index=False)
print("#All  : " + str(len(submission_lgbm)))

print("#True : " + str(len(submission_lgbm[submission_lgbm['target'] > 0.5])))