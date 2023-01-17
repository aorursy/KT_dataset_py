# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import tensorflow
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import timedelta, date
import seaborn as sns
import matplotlib.cm as CM
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,GRU
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import train_test_split



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv',dtype={'fullVisitorId': str})
submission = test_data[['fullVisitorId']].copy()
train_data.head()
test_data.head()
train_data["revenue"] = pd.DataFrame(train_data.totals.apply(json.loads).tolist())[["transactionRevenue"]].fillna(0)

train_data['date'] = train_data['date'].astype('str')
test_data['date'] = test_data['date'].astype('str')

tmp_geo_df_train = pd.DataFrame(train_data.geoNetwork.apply(json.loads).tolist())[["continent","subContinent","country","city"]]
tmp_geo_df_test = pd.DataFrame(test_data.geoNetwork.apply(json.loads).tolist())[["continent","subContinent","country","city"]]
tmp_device_df_train = pd.DataFrame(train_data.device.apply(json.loads).tolist())[["deviceCategory","isMobile"]]
tmp_device_df_test = pd.DataFrame(test_data.device.apply(json.loads).tolist())[["deviceCategory","isMobile"]]

tmp_totals_df_train =  pd.DataFrame(train_data.totals.apply(json.loads).tolist())[['bounces', 'hits', 'newVisits', 'pageviews','visits']]
tmp_totals_df_test =  pd.DataFrame(test_data.totals.apply(json.loads).tolist())[['bounces', 'hits', 'newVisits', 'pageviews','visits']]
tmp_traffic_source_df_train = pd.DataFrame(train_data.trafficSource.apply(json.loads).tolist())[[ 'isTrueDirect', 'medium', 'source']]
tmp_traffic_source_df_test = pd.DataFrame(test_data.trafficSource.apply(json.loads).tolist())[[ 'isTrueDirect', 'medium', 'source']]
train_data.drop(['device','geoNetwork','trafficSource','totals',],axis = 1,inplace=True)
test_data.drop(['device','geoNetwork','trafficSource','totals',],axis = 1,inplace=True)
train_data.head()
train_data  = train_data.join(tmp_geo_df_train,how = "outer")
train_data  = train_data.join(tmp_device_df_train,how = "outer")
train_data  = train_data.join(tmp_traffic_source_df_train,how = "outer")
train_data  = train_data.join(tmp_totals_df_train,how = "outer")
tmp_device_df_train,tmp_geo_df_train,tmp_totals_df_train,tmp_traffic_source_df_train = None,None,None,None
test_data  = test_data.join(tmp_geo_df_test,how = "outer")
test_data  = test_data.join(tmp_device_df_test,how = "outer")
test_data  = test_data.join(tmp_traffic_source_df_test,how = "outer")
test_data  = test_data.join(tmp_totals_df_test,how = "outer")
tmp_device_df_test,tmp_geo_df_test,tmp_totals_df_test,tmp_traffic_source_df_test = None,None,None,None


train_data.head()
train_data.info()
train_data.drop(['date','fullVisitorId','sessionId','visitId','visitStartTime','source','city','country'],axis = 1,inplace=True)
test_data.drop(['date','fullVisitorId','sessionId','visitId','visitStartTime','source','city','country'],axis = 1,inplace=True)

train_data.isTrueDirect.fillna(False,inplace=True)
train_data.newVisits.fillna(0,inplace=True)
train_data.bounces.fillna(0,inplace=True)
train_data.pageviews.fillna(0,inplace=True)
test_data.isTrueDirect.fillna(False,inplace=True)
test_data.newVisits.fillna(0,inplace=True)
test_data.bounces.fillna(0,inplace=True)
test_data.pageviews.fillna(0,inplace=True)
train_data.info()
all_data = pd.concat((train_data,test_data))
categorical_columns = ['channelGrouping','deviceCategory','subContinent','socialEngagementType',
                      'medium','isTrueDirect','isMobile','bounces','newVisits','continent']
for column in categorical_columns:
    train_data[column] = train_data[column].astype('category', categories = all_data[column].unique())
    test_data[column] = test_data[column].astype('category', categories = all_data[column].unique())
train_data = pd.get_dummies(train_data,columns=categorical_columns)
test_data = pd.get_dummies(test_data,columns=categorical_columns)
train_data.shape,test_data.shape
train_data.revenue = train_data.revenue.astype(float)
train_data.revenue = np.log1p(train_data.revenue)
X = train_data.drop('revenue',axis = 1)
Y = train_data.revenue
X.hits = X.hits.astype('uint8')
X.pageviews = X.pageviews.astype('uint8')
X.visits = X.visits.astype('uint8')
test_data.hits = test_data.hits.astype('uint8')
test_data.pageviews = test_data.pageviews.astype('uint8')
test_data.visits = test_data.visits.astype('uint8')


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
model = XGBRegressor(n_estimators=100,
                    learning_rate = .1,
                    max_depth = 6,
                    random_state=42,
                    n_jobs = -1,
                    early_stopping_rounds=10)
model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse",
    eval_set=[(X_test, y_test)],
    verbose=True, )
predictions = model.predict(test_data)
predictions
figsize=(10,10)
fig, ax = plt.subplots(1,1,figsize=figsize)
plot_importance(model, ax=ax,height = 1)
submission['fullVisitorId'] = submission['fullVisitorId']
submission['PredictedLogRevenue'] = predictions
submission["PredictedLogRevenue"] = np.expm1(submission["PredictedLogRevenue"])
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
submission_sum = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
submission_sum["PredictedLogRevenue"] = np.log1p(submission_sum["PredictedLogRevenue"])
submission_sum.to_csv("submission.csv", index=False)


