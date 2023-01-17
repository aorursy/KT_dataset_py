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
train = pd.read_csv("../input/gstoreextractedjson/train_cleaned.csv", 
                    dtype={"fullVisitorId": "str", "sessionId": "str"}, 
                    parse_dates=["date"])
test = pd.read_csv("../input/gstoreextractedjson/test_cleaned.csv", 
                    dtype={"fullVisitorId": "str", "sessionId": "str"}, 
                    parse_dates=["date"])
for i in train.columns:
    if train[str(i)].dtypes == object:
        print(i)
sum(train['totals.pageviews'] == 1), sum(train['totals.transactionRevenue'][train['totals.pageviews'] == 1]) 
sum(train['totals.bounces'] == 1), sum(train['totals.transactionRevenue'][train['totals.bounces'] == 1]) 
idx_tr = train['totals.pageviews'] == 1
train['totals.bounces'][idx_tr] = 1
idx_t = test['totals.pageviews'] == 1
test['totals.bounces'][idx_t] = 1
train['ratioPageHits'] = train['totals.pageviews']/train['totals.hits']
test['ratioPageHits'] = test['totals.pageviews']/test['totals.hits']
train['visitStartTime'] = pd.to_datetime(train['visitStartTime'], unit='s')
test['visitStartTime'] = pd.to_datetime(test['visitStartTime'], unit='s')
train["weekday"] = train["date"].dt.dayofweek
test["weekday"] = test["date"].dt.dayofweek
train["month"] = train["date"].dt.month
test["month"] = test["date"].dt.month
train['sessionHourOfDay'] = train['visitStartTime'].dt.hour
test['sessionHourOfDay'] = test['visitStartTime'].dt.hour
set(train.columns).difference(set(test.columns))
# split the train dataset into train and valid based on time
'''
X_train = train[train["date"]<=pd.Timestamp(2017,5,31)]
X_val = train[train["date"]>pd.Timestamp(2017,5,31)]
y_train = np.log1p(X_train["totals.transactionRevenue"].values)
y_val = np.log1p(X_val["totals.transactionRevenue"].values)
'''
not_train_cl = [
    'fullVisitorId',
    'sessionId', 
    'visitId',
    'visitStartTime',
    'totals.transactionRevenue',
    'transaction',
    'date'
]
features = list(set(train.columns).difference(set(not_train_cl)))
features
target = train.loc[:,'totals.transactionRevenue'].values
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
pred_test[pred_test<0]=0
submission = test[['fullVisitorId']].copy()
submission.loc[:,'PredictedLogRevenue'] = pred_test
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv("submit.csv", index=False)
