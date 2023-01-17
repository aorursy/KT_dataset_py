from bayes_opt import BayesianOptimization
import datetime
import gc
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os
import pandas as pd 
from pandas.io.json import json_normalize
from pandas.core.common import SettingWithCopyWarning
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import sys
import warnings
warnings.filterwarnings('ignore')
gc.enable()
#csv_path differs from paul's original code, just modify if not running in kaggle kernel

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df
#defining time predictors - Added 

def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    df['visitStartTime_'] = pd.to_datetime(df['visitStartTime'],unit="s")
    df['visitStartTime_year'] = df['visitStartTime_'].apply(lambda x: x.year)
    df['visitStartTime_month'] = df['visitStartTime_'].apply(lambda x: x.month)
    df['visitStartTime_day'] = df['visitStartTime_'].apply(lambda x: x.day)
    df['visitStartTime_weekday'] = df['visitStartTime_'].apply(lambda x: x.weekday())
    return df
date_features = [#"year","month","day","weekday",'visitStartTime_year',
    "visitStartTime_month","visitStartTime_day","visitStartTime_weekday"]
train = load_df("../input/train.csv")
test = load_df("../input/test.csv")
for col in test.columns:
    if len(test[col].value_counts()) == 1:
        test.drop(col,inplace=True,axis=1)
for col in train.columns:
    if len(train[col].value_counts()) == 1:
        train.drop(col,inplace=True,axis=1)
##DID NOT REMOVE(NEW)
test = test[test.columns.drop(list(test.filter(regex='trafficSource.adwordsClickInfo')))]
##DID NOT REMOVE(NEW)
train = train[train.columns.drop(list(train.filter(regex='trafficSource.adwordsClickInfo')))]
num_col = ["totals.hits", "totals.pageviews", "visitNumber"]
#FILL NA (NEW)

for col in num_col:
    train[col] = train[col].fillna("0").astype("int32")
    test[col] = test[col].fillna("0").astype("int32")
#####FEATURE ENGINEERING (NEW)

new_features = ["hits_per_pageviews"]     #combining hits and pageviews (see function below)
new_category_features = ["is_high_hits"]
def feature_engineering(df):
    line = 4
    df['hits_per_pageviews'] = (df["totals.hits"]/(df["totals.pageviews"])).apply(lambda x: 0 if np.isinf(x) else x)
    df['is_high_hits'] = np.logical_or(train["totals.hits"]>line,train["totals.pageviews"]>line).astype(np.int32)
#applying change, as well as time features (NEW)

feature_engineering(train)
feature_engineering(test)
add_time_features(train)
_ = add_time_features(test)
#defining categorical features, target, and useless features (NEW)

category_features = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.subContinent",
            "trafficSource.medium", 
            "trafficSource.source",
            ] + date_features
target = 'totals.transactionRevenue'
useless_col = ["trafficSource.adContent",                                                ##useless based on lgb feature importance/unique values/amount of NA
              "trafficSource.adwordsClickInfo.adNetworkType", 
              "trafficSource.adwordsClickInfo.page",
              "trafficSource.adwordsClickInfo.slot",
              "trafficSource.campaign",
              "trafficSource.referralPath",
              'trafficSource.adwordsClickInfo.isVideoAd',
              "trafficSource.adwordsClickInfo.gclId",
              "trafficSource.keyword"]
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]
for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])
#inputting value for na's (NEW)
train[target] = train[target].fillna("0").astype("int32")
def visit24(myTime):
    visitCount=[]
    for time in myTime:
        visitBool = (myTime > time-pd.Timedelta('24 hours')) & (myTime < time)
        visitCount.append(visitBool.sum())
    return(pd.Series(visitCount))

def visit48(myTime):
    visitCount=[]
    for time in myTime:
        visitBool = (myTime > time-pd.Timedelta('48 hours')) & (myTime < time)
        visitCount.append(visitBool.sum())
    return(pd.Series(visitCount))

def visit72(myTime):
    visitCount=[]
    for time in myTime:
        visitBool = (myTime > time-pd.Timedelta('72 hours')) & (myTime < time)
        visitCount.append(visitBool.sum())
    return(pd.Series(visitCount))

def visit168(myTime):
    visitCount=[]
    for time in myTime:
        visitBool = (myTime > time-pd.Timedelta('168 hours')) & (myTime < time)
        visitCount.append(visitBool.sum())
    return(pd.Series(visitCount))

for df in [train, test]:
    df['visitTotal'] = df.groupby('fullVisitorId')['date'].transform('size')
    #df['v24']=df.groupby('fullVisitorId')['date'].transform(visit24)
    #df['v48']=df.groupby('fullVisitorId')['date'].transform(visit48)
    #df['v72']=df.groupby('fullVisitorId')['date'].transform(visit72)
    df['v168']=df.groupby('fullVisitorId')['date'].transform(visit168)

#taking cleaned data from paul's code and modifying features to improve relevancy (NEW)
all_features = category_features+num_col+new_features+new_category_features
#prepare for lgb

train_x = train[all_features]
train_y = train[target]
test_x = test[all_features]
for col in category_features:
    print("transform column {}".format(col))
    lbe = LabelEncoder()
    lbe.fit(pd.concat([train[col],test_x[col]]).astype("str"))
    train_x[col] = lbe.transform(train_x[col].astype("str"))
    test_x[col] = lbe.transform(test_x[col].astype("str"))
#defining functions

def lgb_eval(num_leaves,max_depth,lambda_l2,lambda_l1,min_child_samples,learning_rate):
    params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : int(num_leaves),
    "max_depth" : int(max_depth),
    "lambda_l2" : lambda_l2,
    "lambda_l1" : lambda_l1,
    "num_threads" : 4,
    "min_child_samples" : int(min_child_samples),
    "learning_rate" : learning_rate,
    "subsample_freq" : 5,
    "bagging_seed" : 42,
    "verbosity" : -1
    }
    lgtrain = lgb.Dataset(train_x, label=np.log1p(train_y.apply(lambda x : 0 if x < 0 else x)),categorical_feature=category_features)
    cv_result = lgb.cv(params,
                       lgtrain,
                       10000,
                       categorical_feature=category_features,
                       early_stopping_rounds=100,
                       stratified=False,
                       nfold=5)
    return -cv_result['rmse-mean'][-1]

def lgb_train(num_leaves,max_depth,lambda_l2,lambda_l1,min_child_samples, learning_rate):
    params = {
    "objective" : "regression",
    "metric" : "rmse", 
    "num_leaves" : int(num_leaves),
    "max_depth" : int(max_depth),
    "lambda_l2" : lambda_l2,
    "lambda_l1" : lambda_l1,
    "num_threads" : 4,
    "min_child_samples" : int(min_child_samples),
    "learning_rate" : learning_rate,
    "subsample_freq" : 5,
    "bagging_seed" : 42,
    "verbosity" : -1
    }
    t_x,v_x,t_y,v_y = train_test_split(train_x,train_y,test_size=0.2)
    lgtrain = lgb.Dataset(t_x, label=np.log1p(t_y.apply(lambda x : 0 if x < 0 else x)),categorical_feature=category_features)
    lgvalid = lgb.Dataset(v_x, label=np.log1p(v_y.apply(lambda x : 0 if x < 0 else x)),categorical_feature=category_features)
    model = lgb.train(params, lgtrain, 2000, valid_sets=[lgvalid], early_stopping_rounds=100, verbose_eval=100)
    pred_test_y = model.predict(test_x, num_iteration=model.best_iteration)
    return pred_test_y, model

#param_tuning defined above, (init_points, num_iter,**args)...utilizes bayesian optimization
##defining bo ranges for parameter tuning    
# def param_tuning(init_points,num_iter,**args):
#     lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (25, 50),
#                                                 'max_depth': (5, 15),
#                                                 'lambda_l2': (0.0, 0.05),
#                                                 'lambda_l1': (0.0, 0.05),
#                                                 'bagging_fraction': (0.5, 0.8),
#                                                 'feature_fraction': (0.5, 0.8),
#                                                 'min_child_samples': (20, 50),
#                                                 })

#     lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
#     return lgbBO
# result = param_tuning(5,15)
def param_tuning_boundaryShift(init_points,num_iter,**args):
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (20, 40),
                                                'max_depth': (14, 30),
                                                'lambda_l2': (0.0, 0.05),
                                                'lambda_l1': (0.0, 0.05),
                                                'min_child_samples': (15, 40), 'learning_rate': (0.01, 0.04)                                            
                                                })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
    return lgbBO
result = param_tuning_boundaryShift(20,10)
result.res['max']['max_params']
pred1,model1 = lgb_train(**result.res['max']['max_params'])
pred2,model2 = lgb_train(**result.res['max']['max_params'])
pred3,model3 = lgb_train(**result.res['max']['max_params'])

