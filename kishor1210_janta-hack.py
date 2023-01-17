# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai import *
from fastai.tabular import *

import time
train = pd.read_csv("/kaggle/input/electricityconsumptions/datasets/train.csv")
test = pd.read_csv("/kaggle/input/electricityconsumptions/datasets/test.csv")

def var2(x):
    if x=='A':
        return 1
    elif x=='B':
        return 2
    else:
        return 3
    
#DayNight
def DayNight(x):
    time_struct = time.strptime(x, '%Y-%m-%d %H:%M:%S')
    hour =int(time.strftime('%H', time_struct))
    if hour > 5 and hour <9:
        return 1
    elif hour > 9 and hour <18:
        return 0
    elif hour >18 and hour<23:
        return 2
    else:
        return 3
#hours  
def hours(x):
    time_struct = time.strptime(x, '%Y-%m-%d %H:%M:%S')
    hour =int(time.strftime('%H', time_struct))
    return int(hour)

#days
def days(x):
    time_struct = time.strptime(x, '%Y-%m-%d %H:%M:%S')
    day = time.strftime('%d', time_struct)
    return int(day)

#Months
def months(x):
    time_struct = time.strptime(x, '%Y-%m-%d %H:%M:%S')
    month = time.strftime('%m', time_struct)#
    return int(month)

#Years
def years(x):
    time_struct = time.strptime(x, '%Y-%m-%d %H:%M:%S')
    year = time.strftime('%Y', time_struct)
    return 2018 - int(year)

#Weekdays
def weekdays(x):
    time_struct = time.strptime(x, '%Y-%m-%d %H:%M:%S')
    week = time.strftime('%u', time_struct)
    return int(week)

#Weekdays
def yeardays(x):
    time_struct = time.strptime(x, '%Y-%m-%d %H:%M:%S')
    yearday = time.strftime('%u', time_struct)
    return int(yearday)

    
train["hours"] = train["datetime"].apply(hours)
test["hours"] = test["datetime"].apply(hours)

train["days"] = train["datetime"].apply(days)
test["days"] = test["datetime"].apply(days)

train["months"] = train["datetime"].apply(months)
test["months"] = test["datetime"].apply(months)

train["years"] = train["datetime"].apply(years)
test["years"] = test["datetime"].apply(years)

#train["weekdays"] = train["datetime"].apply(weekdays)
#test["weekdays"] = test["datetime"].apply(weekdays)

#train["yeardays"] = train["datetime"].apply(yeardays)
#test["yeardays"] = test["datetime"].apply(yeardays)
    
#train["DayNight"] = train["datetime"].apply(DayNight)
#test["DayNight"] = test["datetime"].apply(DayNight)

#train["var2"] = train["var2"].apply(var2)
#test["var2"]=test["var2"].apply(var2)
    

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])


def var2(x):
    if x=='A':
        return 1
    elif x=='B':
        return 2
    else:
        return 3
    
def booleancon(x):
    if x == True:
        return 1
    else:
        return 0
    
def time_pr(train):
    train = add_datepart(train,'datetime',drop=True,time=True)
    train['datetimeIs_month_end'] = train['datetimeIs_month_end'].apply(booleancon)
    train['datetimeIs_month_start']   = train['datetimeIs_month_start'].apply(booleancon)
    train['datetimeIs_quarter_start'] = train['datetimeIs_quarter_start'].apply(booleancon)
    train['datetimeIs_quarter_end'] = train['datetimeIs_quarter_end'].apply(booleancon)
    train['datetimeIs_year_start'] = train['datetimeIs_quarter_start'].apply(booleancon)
    train['datetimeIs_year_end'] = train['datetimeIs_quarter_end'].apply(booleancon)
    train['var2'] = train['var2'].apply(var2)
    return train


pd.set_option('display.max_columns', 1000)  # or 1000
pd.set_option('display.max_rows', 1000)  # or 1000
pd.set_option('display.max_colwidth', 199)  # or 199
train = time_pr(train)
test  = time_pr(test)
#train["YearFar"] = 2018 - train["datetimeYear"]
#test["YearFar"] = 2018 - test["datetimeYear"]
train.head()
dummy_train = train[train['datetimeDay']<=16]
dummy_test = train[train['datetimeDay']>16]
col = []
for i in train.columns:
    if i!= 'electricity_consumption' and i!='ID' and i!='datetime':
        col.append(i)
X = train[col].values
Y = train['electricity_consumption'].values
x_train = dummy_train[col].values
y_train = dummy_train['electricity_consumption'].values
x_test = dummy_test[col].values
y_test = dummy_test['electricity_consumption'].values

import lightgbm as lgb

d_train = lgb.Dataset(x_train, label=y_train)
d_test = lgb.Dataset(x_test, label=y_test)


params = {}
params['application']='root_mean_squared_error'
params['num_boost_round'] = 1000
params['learning_rate'] = 0.02
params['boosting_type'] = 'gbdt'
params['metric'] = 'rmse'
params['sub_feature'] = 0.833
params['num_leaves'] = 15
params['min_split_gain'] = 0.05
params['min_child_weight'] = 27
params['max_depth'] = -1
params['num_threads'] = 10
params['max_bin'] = 50
params['lambda_l2'] = 0.10
params['lambda_l1'] = 0.30
params['feature_fraction']= 0.833
params['bagging_fraction']= 0.979
clf = lgb.train(params, d_train, 2000,d_test,verbose_eval=200, early_stopping_rounds=200)
#RandomForest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=250)
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmae = sqrt(mean_squared_error(pred,y_test))
rmae
#Xgboost
from xgboost import XGBRegressor
p=XGBRegressor(n_estimators=30000,random_state=1729,learning_rate=0.017,max_depth=4,n_jobs=4)
# max_depth=5,0.018
p.fit(x_train,y_train,eval_set=[(x_test, y_test)],eval_metric='rmse',early_stopping_rounds=500,verbose=200)
#Catboost
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(n_estimators = 1000,
    loss_function = 'RMSE',
    eval_metric = 'RMSE')
cb_model.fit(x_train, y_train, use_best_model=True, eval_set=(x_test, y_test), early_stopping_rounds=50)
d_train = lgb.Dataset(X, label=Y)
params = {}
params['application']='root_mean_squared_error'
params['num_boost_round'] = 1000
params['learning_rate'] = 0.02
params['boosting_type'] = 'gbdt'
params['metric'] = 'rmse'
params['sub_feature'] = 0.833
params['num_leaves'] = 15
params['min_split_gain'] = 0.05
params['min_child_weight'] = 27
params['max_depth'] = -1
params['num_threads'] = 10
params['max_bin'] = 50
params['lambda_l2'] = 0.10
params['lambda_l1'] = 0.30
params['feature_fraction']= 0.833
params['bagging_fraction']= 0.979
clf = lgb.train(params, d_train, 2000)
x_test = test[col]
pred = clf.predict(x_test)
pred
test['electricity_consumption'] = pred

test.head()
test[['ID','electricity_consumption']].to_csv('submission.csv',header=True,index = None)
col_dict ={}
z=0
for i in col:
    col_dict[i]=z
    z=z+1
    
col_dict
lgb.plot_importance(clf,importance_type='split', max_num_features=22)

