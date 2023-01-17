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
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    
    return df
start_time = time.time()

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print('[{}] Finished to load data'.format(time.time() - start_time))
train = dataPreProcessTime(train)
test = dataPreProcessTime(test)

y = train['is_attributed']
train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)
# from imblearn.combine import SMOTETomek
# os_us = SMOTETomek()
# train, y = os_us.fit_sample(train, y)
from sklearn import preprocessing
normalizer = preprocessing.Normalizer().fit(train)
train = normalizer.transform(train)
test = normalizer.transform(test)

print('[{}] Start XGBoost Training'.format(time.time() - start_time))

params = {'eta': 0.17,
		  'max_depth': 10,
		  'subsample': 0.9,
		  'colsample_bytree': 1.0,
		  'colsample_bylevel': 0.7,
		  'min_child_weight': 10,
		  'alpha': 4,
		  'objective': 'binary:logistic',
		  'eval_metric': 'auc',
		  'random_state': 100,
		  'silent': False}
          
x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=100)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 300, watchlist, maximize=True, verbose_eval=10)

print('[{}] Finish XGBoost Training'.format(time.time() - start_time))
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('sub_xg.csv',index=False)