import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime
import calendar
from math import sin, cos, sqrt, atan2, radians,asin

from datetime import timedelta
import datetime as dt
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)
plt.style.use('fivethirtyeight')
import folium
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import operator
import pickle
import os
input_path='../input/eda-and-feature-engineering//'
train=pd.read_csv(input_path+'train_cleaned.csv')
test=pd.read_csv(input_path+'test_cleaned.csv')
print("Shape of Training Data ",train.shape)
print("Shape of Testing Data ",test.shape)
drop_columns=['key','pickup_datetime','pickup_date','pickup_latitude_round3','pickup_longitude_round3','dropoff_latitude_round3','dropoff_longitude_round3']
train_1=train.drop(drop_columns,axis=1)
test_1=test.drop(drop_columns,axis=1)
print("Shape of Training Data after dropping columns",train_1.shape)
print("Shape of Testing Data after dropping columns",test_1.shape)
train_1=pd.get_dummies(train_1)
test_1=pd.get_dummies(test_1)

print("Shape of Training Data after One Hot Encoding",train_1.shape)
print("Shape of Testing Data after One Hot Encoding",test_1.shape)
X=train_1.drop(['fare_amount'],axis=1)
y=train_1['fare_amount']

#split data into train and validation data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print("Number of records in training data ",X_train.shape[0])
print("Number of records in validation data ",X_test.shape[0])
lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred=lm.predict(X_test)
lm_rmse=np.sqrt(mean_squared_error(y_pred, y_test))
print("RMSE for Linear Regression is ",lm_rmse)
linear_reg_pred=lm.predict(test_1)

submissions=pd.read_csv('../input/new-york-city-taxi-fare-prediction/sample_submission.csv')
submissions['fare_amount']=linear_reg_pred
submissions.to_csv("LinearRegression_Baseline.csv",index=False)
def XGBoost(X_train,X_test,y_train,y_test,num_rounds=300):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=num_rounds, 
                    early_stopping_rounds=20,evals=[(dtest,'test')],)
xgbm = XGBoost(X_train,X_test,y_train,y_test)
xgbm_pred = xgbm.predict(xgb.DMatrix(test_1), ntree_limit = xgbm.best_ntree_limit)
submissions['fare_amount']=xgbm_pred
submissions.to_csv("XGboost_Baseline.csv",index=False)
importance=xgbm.get_score()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'score'])
plt.figure()

df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(10, 25))
plt.title("Feature Importance")
del train_1
del test_1
del X_train,X_test,y_train,y_test
lgr=(-73.8733, 40.7746)
jfk=(-73.7900, 40.6437)
ewr=(-74.1843, 40.6924)


def distance(lat1,lon1,lat2,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

test['pickup_distance_jfk']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],jfk[1],jfk[0]),axis=1)
test['dropoff_distance_jfk']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],jfk[1],jfk[0]),axis=1)
test['pickup_distance_ewr']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],ewr[1],ewr[0]),axis=1)
test['dropoff_distance_ewr']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],ewr[1],ewr[0]),axis=1)
test['pickup_distance_laguardia']=test.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],lgr[1],lgr[0]),axis=1)
test['dropoff_distance_laguardia']=test.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],lgr[1],lgr[0]),axis=1)



train['pickup_distance_jfk']=train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],jfk[1],jfk[0]),axis=1)
train['dropoff_distance_jfk']=train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],jfk[1],jfk[0]),axis=1)
train['pickup_distance_ewr']=train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],ewr[1],ewr[0]),axis=1)
train['dropoff_distance_ewr']=train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],ewr[1],ewr[0]),axis=1)
train['pickup_distance_laguardia']=train.apply(lambda row:distance(row['pickup_latitude'],row['pickup_longitude'],lgr[1],lgr[0]),axis=1)
train['dropoff_distance_laguardia']=train.apply(lambda row:distance(row['dropoff_latitude'],row['dropoff_longitude'],lgr[1],lgr[0]),axis=1)


train_1=train.drop(drop_columns,axis=1)
test_1=test.drop(drop_columns,axis=1)
print("Shape of Training Data after dropping columns",train_1.shape)
print("Shape of Testing Data after dropping columns",test_1.shape)


train_1=pd.get_dummies(train_1)
test_1=pd.get_dummies(test_1)

print("Shape of Training Data after One Hot Encoding",train_1.shape)
print("Shape of Testing Data after One Hot Encoding",test_1.shape)

X=train_1.drop(['fare_amount'],axis=1)
y=train_1['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print("Number of records in training data ",X_train.shape[0])
print("Number of records in validation data ",X_test.shape[0])

xgbm = XGBoost(X_train,X_test,y_train,y_test,num_rounds=1500)
xgbm_pred = xgbm.predict(xgb.DMatrix(test_1), ntree_limit = xgbm.best_ntree_limit)

submissions['fare_amount']=xgbm_pred
submissions.to_csv("XGboost_WithDistancetoAirport.csv",index=False)

importance=xgbm.get_score()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'score'])
plt.figure()

df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(10, 25))
plt.title("Feature Importance")

train.to_csv("train_cleaned.csv",index=False)
test.to_csv("test_cleaned.csv",index=False)