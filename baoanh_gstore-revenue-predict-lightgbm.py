import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

import seaborn as sns
sns.set()

import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
os.listdir("../input/gstoreextractedjson/")
train = pd.read_csv("../input/gstoreextractedjson/train_cleaned.csv", 
                    dtype={"fullVisitorId": "str", "sessionId": "str"}, 
                    parse_dates=["date"])
test = pd.read_csv("../input/gstoreextractedjson/test_cleaned.csv", 
                    dtype={"fullVisitorId": "str", "sessionId": "str"}, 
                    parse_dates=["date"])
train.shape, test.shape
target = train.loc[:,'totals.transactionRevenue'].values
del train["totals.transactionRevenue"]
train.shape, test.shape, target.shape
num_cols = ['totals.bounces', 
            'totals.hits', 
            'totals.newVisits', 
            'totals.pageviews']
cat_cols = ['channelGrouping', 
            'device.browser', 
            'device.deviceCategory', 
            'device.operatingSystem', 
            'geoNetwork.city', 
            'geoNetwork.continent', 
            'geoNetwork.country', 
            'geoNetwork.metro', 
            'geoNetwork.networkDomain', 
            'geoNetwork.region', 
            'geoNetwork.subContinent', 
            'trafficSource.adContent', 
            'trafficSource.adwordsClickInfo.adNetworkType', 
            'trafficSource.adwordsClickInfo.gclId', 
            'trafficSource.adwordsClickInfo.slot', 
            'trafficSource.campaign', 
            'trafficSource.keyword', 
            'trafficSource.medium', 
            'trafficSource.referralPath', 
            'trafficSource.source', 
            'device.isMobile', 
            'trafficSource.adwordsClickInfo.isVideoAd', 
            'trafficSource.adwordsClickInfo.page', 
            'trafficSource.isTrueDirect']
features = num_cols+cat_cols
X_train, X_val, y_train, y_val = train_test_split(train[features], target, test_size=0.2, random_state=42)
def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)

def run_lgb(X_train, y_train, X_val, y_val, X_test):
    data_train = lgb.Dataset(X_train, label=y_train)
    data_val = lgb.Dataset(X_val, label=y_val)
    param = {
         'objective':'regression',
         'metric': 'rmse',
         'learning_rate':0.005,
         'num_leaves':40,
         'min_data_in_leaf':150,
         'max_depth':10,
         'bagging_fraction':0.6,
         'feature_fraction':0.6,
         'bagging_frequency': 6,
         'verbosity':-1,
         'random_state': 42}
    model = lgb.train(param, data_train, valid_sets=[data_train, data_val], num_boost_round=5000, early_stopping_rounds=100,
                  verbose_eval=200)
    pred_y_test = model.predict(X_test, num_iteration=model.best_iteration)
    
    return pred_y_test, model
pred_test, model = run_lgb(X_train, y_train, X_val, y_val, test[features])
pred_test = model.predict(test[num_cols + cat_cols], num_iteration=model.best_iteration)
pred_test[pred_test<0]=0
submission = test[['fullVisitorId']].copy()
submission.loc[:,'PredictedLogRevenue'] = pred_test
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv("submit.csv", index=False)
