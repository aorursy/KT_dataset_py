import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from scipy import stats
import lightgbm as lgb
import json
import gc
from pandas.io.json import json_normalize
import time
testrows = None
#based in SWK https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
def load_df(csv_path='../input/train.csv', nrows=None,isTest=False):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    if isTest == False:
        df['totals.transactionRevenue'].fillna(0,inplace=True)
        df['totals.transactionRevenue'] = df['totals.transactionRevenue'].astype('float')
    df['totals.pageviews'].fillna(0,inplace=True)
    df['totals.pageviews'] = df['totals.pageviews'].astype('int64')
    df['totals.hits'].fillna(0,inplace=True)
    df['totals.hits'] = df['totals.hits'].astype('int64')
    df['fullVisitorId'] = df['fullVisitorId'].astype('str')
    return df
train = load_df(nrows=testrows)
# list_device = ['device.browser','device.operatingSystem','device.deviceCategory','geoNetwork.country']
# for device_feature in list_device:
#     agg = train.groupby(['fullVisitorId'])[device_feature].size().reset_index(name=device_feature+'_size')
#     train = train.merge(agg,how='left',on=['fullVisitorId'])
train.info()
def getHour(timestamp):
    struct_time = time.gmtime(timestamp)
    return struct_time.tm_hour

def getMinutes(timestamp):
    struct_time = time.gmtime(timestamp)
    return struct_time.tm_min

def getSeconds(timestamp):
    struct_time = time.gmtime(timestamp)
    return struct_time.tm_sec

def getWeekDay(timestamp):
    struct_time = time.gmtime(timestamp)
    return struct_time.tm_wday

def getMonth(timestamp):
    struct_time = time.gmtime(timestamp)
    return struct_time.tm_mon

train['visitStartMonth'] = train['visitStartTime'].apply(lambda x:getMonth(x))
train['visitStartWeekDay'] = train['visitStartTime'].apply(lambda x:getWeekDay(x))
train['visitStartTimeHour'] = train['visitStartTime'].apply(lambda x:getHour(x))
train['visitStartTimeMinute'] = train['visitStartTime'].apply(lambda x:getMinutes(x))
train['visitStartTimeHourRegion'] =  train['geoNetwork.country']+train['visitStartTimeHour'].astype('str')+train['geoNetwork.city']
train['pageviewshigh'] = train['totals.pageviews'] > 8
train['pagehitshigh'] = train['totals.hits'] > 8
def convert_category(train):
    for index in train:
        if train[index].dtype == 'object' and index != 'fullVisitorId':
            le =  preprocessing.LabelEncoder()
            train_val = list(train[index].values.astype(str))
            le.fit(list(train[index].values.astype(str)))
            train[index] = le.transform(train_val)
convert_category(train)

train.corr()['totals.transactionRevenue'].sort_values()
def preprocess(x):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False).fit(x)
    scaler.transform(x)
    return x

x = preprocess(train.drop(['fullVisitorId','totals.transactionRevenue'],axis=1))
y = np.log1p(train['totals.transactionRevenue'])
del train
gc.collect()
d_train = lgb.Dataset(x, y)
iterations = 8000
watchlist = [d_train]
#parameters based on https://www.kaggle.com/pavansanagapati/simple-exploration-lgbm-model-lb-1-4221
params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 50, "learning_rate" : 0.02, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9}

model = lgb.train(params, train_set=d_train, num_boost_round=iterations, 
        valid_sets=watchlist, early_stopping_rounds=150, verbose_eval=200) 
test = load_df('../input/test.csv',nrows=testrows,isTest=True)
fullVisitor = test[['fullVisitorId']]
convert_category(test)
x_test = preprocess(test.drop(['fullVisitorId'],axis=1))
del test
gc.collect()
predict = model.predict(x_test)
predict[predict <= 0.1] = 0
fullVisitor['PredictedLogRevenue'] = np.expm1(predict)
fullVisitor = fullVisitor.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
fullVisitor.columns = ["fullVisitorId", "PredictedLogRevenue"]
fullVisitor["PredictedLogRevenue"] = np.log1p(fullVisitor["PredictedLogRevenue"])
fullVisitor[['fullVisitorId','PredictedLogRevenue']].to_csv("baseline_lgb.csv", index = False)
fullVisitor[['fullVisitorId','PredictedLogRevenue']]