# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import pickle

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import gc

from sklearn.model_selection import train_test_split

from sklearn import model_selection



import lightgbm as lgb

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_squared_error, roc_auc_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/citizensdata/Citizens"))



path = "../input/citizensdata/Citizens"

# Any results you write to the current directory are saved as output.
#import datetime

#Read in the data

test = open(path+'/datathon_propattributes.obj', 'rb')

test = pickle.load(test)



checkdate = '2018-10-01'

checkdate = pd.to_datetime(checkdate)

testset = test[test['transaction_dt'] >= checkdate]

trainset = test[test['transaction_dt'] < checkdate]



pd.set_option('display.max_rows', 200)

#test.dtypes
col_use = ['prop_state', 'prop_zip_code','construction_quality', 'roof_cover','condition','dwelling_type','acres','assessed_total_value','assessed_land_value','market_total_value','market_improvement_value','sale_amt']

train = pd.DataFrame(trainset[col_use])

test = pd.DataFrame(testset[col_use])
def var2num(test,var,n_label=0):

    if n_label==0:

        col = test[var].value_counts().index.values.tolist()

    else:

        col = test[var].value_counts().index[:n_label].values.tolist()

    ind = [i+1 for i in range(len(col))]

    Zip = zip(col,ind)

    dic = dict(Zip)

    new = test[var].apply(lambda x:dic[x] if x in dic else None)

    mean = new.mean()

    new_col = new.fillna(mean)

    return new_col
col_trim = ['sale_amt']
train = train[(np.abs(stats.zscore(train[col_trim])) < 3).all(axis=1)]
var = 'prop_state'

train[var] = var2num(train,var,n_label=0)

test[var] = var2num(test,var,n_label=0)
var = 'prop_zip_code'

train[var] = var2num(train,var,n_label=50)

test[var] = var2num(test,var,n_label=50)
var = 'construction_quality'

train[var] = var2num(train,var,n_label=0)

test[var] = var2num(test,var,n_label=0)

var = 'roof_cover'

train[var] = var2num(train,var,n_label=10)

test[var] = var2num(test,var,n_label=10)

var = 'condition'

train[var] = var2num(train,var,n_label=0)

test[var] = var2num(test,var,n_label=0)

var = 'dwelling_type'

train[var] = var2num(train,var,n_label=10)

test[var] = var2num(test,var,n_label=10)
train.describe()
col_X = ['prop_state', 'prop_zip_code','construction_quality', 'roof_cover','condition','dwelling_type','acres','assessed_total_value','assessed_land_value','market_total_value','market_improvement_value']

train_X = train[col_X]

train_Y = train['sale_amt']

train_X = (train_X-train_X.mean())/(train_X.max()-train_X.min())

test_X = test[col_X]

test_Y = test['sale_amt']

test_X = (test_X-test_X.mean())/(test_X.max()-test_X.min())
gc.collect()
def run_lgb(train_X, train_y, val_X, val_y, test_X):

    params = {

        "objective" : "regression",

        "metric" : "rmse",

        'boosting_type' : 'goss',

        'max_depth' : 5,#-1

        "num_leaves" : 20,#20

        "learning_rate" : 0.01,#0.01

        #"bagging_fraction" : 0.6,#0.7 #0.8 #0.3

        "feature_fraction" : 0.6,#0.7 #0.5

        #"bagging_freq" : 2, #10 #20

        "bagging_seed" : 42, #2018

        "verbosity" : -1,

        'lambda_l2' : 0.000001,#0.1

        'lambda_l1' : 0.00001,#0,

        'max_bin' : 200 #default=250 #200 #170 #120 #90



    }

    

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    evals_result = {}

    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, 

                      verbose_eval=200, evals_result=evals_result)

    

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result
# X_train, X_val, y_train, y_val = train_test_split( train_X.values, train_Y.values, test_size=0.1, random_state=2019)
# seeds = [42]

# pred_test_full_seed = 0

# for seed in seeds:

#     kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)

#     pred_test_full = 0

#     for dev_index, val_index in kf.split(X_train):

#         dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]

#         dev_y, val_y = train_Y[dev_index], train_Y[val_index]

#         pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

#         pred_test_full += pred_test

#     pred_test_full /= 5.

#     pred_test_full = np.expm1(pred_test_full)

#     pred_test_full_seed += pred_test_full

#     print("Seed {} completed....".format(seed))

# pred_test_full_seed /= np.float(len(seeds))



# print("LightGBM Training Completed...")