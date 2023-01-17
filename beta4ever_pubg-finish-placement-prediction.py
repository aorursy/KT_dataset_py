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
# import libaries
import matplotlib.pyplot as plt
import seaborn as sns

# show results in notebook
%matplotlib inline

# import data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
    
# 完成
print("train data: {} in total; data fields: {}; ".format(*train_df.shape))
print(train_df.columns)
print("test data: {} in total; data fields: {}; ".format(*test_df.shape))
print(test_df.columns)
# data exploring
train_df.info(null_counts=True)
print("----------------------------------------")
test_df.info(null_counts=True)
def feature_engineering(df):
    df['landDistance']=df['rideDistance']+df['walkDistance']
    return None

feature_engineering(train_df)
feature_engineering(test_df)

train_df.head(5)
#train result split
train_results = train_df['winPlacePerc']
train_features_raw = train_df.drop(['winPlacePerc'],axis=1)


#test dataset 
test_id = test_df['Id']
# feature selection
train_features = train_features_raw.drop(['Id','groupId','matchId','killStreaks','numGroups','rideDistance','walkDistance'],axis=1)
test_features = test_df.drop(['Id','groupId','matchId','killStreaks','numGroups','rideDistance','walkDistance'],axis=1)
# train valid split by using train_test_split
from sklearn.model_selection import train_test_split

tsize = 0.1

X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_results, test_size = tsize, shuffle=True)
# using lightGBM
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid,label=y_valid,reference=lgb_train)
# evaluation function
def mae_metric(yhat,ytrain):
    y = ytrain.get_label()
    return 'mae', np.mean(abs(yhat-y)), False
# model params
params = {
#    'device': 'gpu',
    'objective': 'regression', 
    'boosting': 'gbdt', #gbdt, rf, dart, goss
    'learning_rate': 0.01, 
#    'num_leaves': 2**14, # num_leaves = 2^(max_depth) #31
    'tree' : 'data', #serial, feature, data
#    'max_depth' : -1,
#    'min_data_in_leaf' : 10,
#    'min_sum_hessian_in_leaf' : 1,
    'feature_fraction': 0.8,
#    'bagging_fraction': 0.8, 
#    'bagging_freq': 1, 
#    'reg_alpha' : 0.1, 
    'reg_lambda' : 1, 
#    'min_split_gain' : 0.1, 
}

num_rounds = 10000
stop_rounds = 1000
import time

starttime = time.time()

bst = lgb.train(params, lgb_train, num_rounds, valid_sets=lgb_valid, early_stopping_rounds=stop_rounds, feval=mae_metric)

print("bst.best_iteration",bst.best_iteration)

print('time consumed（mins）：' + str((time.time() - starttime)/60))

y_test_predict = bst.predict(test_features, num_iteration=bst.best_iteration)
submission = pd.Series()
submission = submission.append(pd.Series(y_test_predict, index=test_id))

## save submission
submission = pd.DataFrame({ "Id": submission.index, "winPlacePerc": submission.values})
submission.to_csv('submission.csv', index=False)