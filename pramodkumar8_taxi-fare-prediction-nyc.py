# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", nrows = 2000000, parse_dates= ["key","pickup_datetime"])
test = pd.read_csv("../input/test.csv", parse_dates= ["key","pickup_datetime"])
train.head()
train.shape
test.shape
train.head(10)
train.describe()
#check for missing values in train data
train.isnull().sum().sort_values(ascending=False)
#check for missing values in test data
test.isnull().sum().sort_values(ascending=False)
#train.hist(bins = 100, figsize=(15,12))
train.plot(kind = "scatter", x="dropoff_latitude",y="dropoff_longitude")
train.plot(kind = "scatter", x="pickup_latitude",y="pickup_longitude")
train[train.isnull().any(1)].index
#drop the missing values
train = train.drop(train[train.isnull().any(1)].index, axis = 0)
train.shape
#check the target column
train['fare_amount'].describe()
#420 fields have negative fare_amount values.
from collections import Counter
Counter(train['fare_amount']<0)
train = train.drop(train[train['fare_amount']<0].index, axis=0)
train.shape
#no more negative values in the fare field
train['fare_amount'].describe()
#highest fare is $500
train['fare_amount'].sort_values(ascending=False)
train[train["fare_amount"] >= 500].index
train = train.drop(train[train["fare_amount"] >= 500].index, axis = 0)
train['passenger_count'].describe()
#max is 208 passengers. Assuming that a bus is a 'taxi' in NYC, I don't think a bus can carry 208 passengers! Let' see the distribution of this field
#LOL! One field. this is DEFINITELY an outlier. Lets drop it 
train[train['passenger_count']>6]
train = train.drop(train[train['passenger_count']>6].index, axis = 0)
#much neater now! Max number of passengers are 6. Which makes sense is the cab is an SUV :)
train['passenger_count'].describe()
#Next, let us explore the pickup latitude and longitudes
train['pickup_latitude'].describe()
train.describe()

sns.distplot(train.dropoff_longitude)
plat = np.percentile(train.pickup_latitude,[2,99.9])
plat
dlat = np.percentile(train.dropoff_latitude,[2,99.9])
dlat
plon = np.percentile(train.pickup_longitude,[1,98])
plon
dlon = np.percentile(train.dropoff_longitude,[1,98])
dlon
train.shape
train = train[(train.pickup_latitude > 39) & (train.pickup_latitude < 42)
              &(train.pickup_longitude > -75) & (train.pickup_longitude < -72)
              &(train.dropoff_longitude > -75) & (train.dropoff_longitude < -72)
              &(train.dropoff_latitude > 39) & (train.dropoff_latitude < 42)]
train.shape
train.plot(kind = "scatter", x="pickup_latitude",y="pickup_longitude", alpha = 0.1)
train.plot(kind = "scatter", x = "dropoff_latitude", y="dropoff_longitude", alpha = 0.1)
def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        R = 6371  #radius of earth in kilometers
        #R = 3959 #radius of earth in miles
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        i['H_Distance'] = d
    return d
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
train['H_Distance'].head(10)
train.head(10)
data = [train,test]
for i in data:
    i['Year'] = i['pickup_datetime'].dt.year
    i['Month'] = i['pickup_datetime'].dt.month
    i['Date'] = i['pickup_datetime'].dt.day
    i['Day of Week'] = i['pickup_datetime'].dt.dayofweek
    i['Hour'] = i['pickup_datetime'].dt.hour
train.head()
test.head()
plt.figure(figsize=(15,7))
plt.hist(train['passenger_count'], bins=15)
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')
plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=1.5)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.figure(figsize=(15,7))
plt.hist(train['Hour'], bins=100)
plt.xlabel('Hour')
plt.ylabel('Frequency')
plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')
plt.figure(figsize=(15,7))
plt.hist(train['Day of Week'], bins=100)
plt.xlabel('Day of Week')
plt.ylabel('Frequency')
plt.figure(figsize=(15,7))
plt.scatter(x=train['Day of Week'], y=train['fare_amount'], s=1.5)
plt.xlabel('Day of Week')
plt.ylabel('Fare')
train.sort_values(['H_Distance','fare_amount'], ascending=False)
len(train)
train.loc[(train['H_Distance'] == 0), ['H_Distance']]
bins_0 = train.loc[(train['H_Distance'] == 0), ['H_Distance']]
bins_1 = train.loc[(train['H_Distance'] > 0) & (train['H_Distance'] <= 10),['H_Distance']]
bins_2 = train.loc[(train['H_Distance'] > 10) & (train['H_Distance'] <= 50),['H_Distance']]
bins_3 = train.loc[(train['H_Distance'] > 50) & (train['H_Distance'] <= 100),['H_Distance']]
bins_4 = train.loc[(train['H_Distance'] > 100) & (train['H_Distance'] <= 200),['H_Distance']]
bins_5 = train.loc[(train['H_Distance'] > 200) & (train['H_Distance'] <= 300),['H_Distance']]
bins_6 = train.loc[(train['H_Distance'] > 300),['H_Distance']]
bins_0['bins'] = '0'
bins_1['bins'] = '0-10'
bins_2['bins'] = '11-50'
bins_3['bins'] = '51-100'
bins_4['bins'] = '100-200'
bins_5['bins'] = '201-300'
bins_6['bins'] = '>300'
dist_bins =pd.concat([bins_0,bins_1,bins_2,bins_3,bins_4,bins_5,bins_6])
#len(dist_bins)
dist_bins.columns
train.loc[(train['H_Distance'] > 100) & (train['H_Distance'] <= 500),['H_Distance']]
Counter(dist_bins['bins'])
#pickup latitude and longitude = 0
train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))&((train['dropoff_latitude']!=0) & (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)]
train = train.drop(train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))&((train['dropoff_latitude']!=0) & (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)].index, axis=0)
#1 row dropped
train.shape
#Check in test data
test.loc[((test['pickup_latitude']==0) & (test['pickup_longitude']==0))&((test['dropoff_latitude']!=0) & (test['dropoff_longitude']!=0))]
#No records! PHEW!
#dropoff latitude and longitude = 0
train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0)) & (train['fare_amount']==0)]
train = train.drop(train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0)) & (train['fare_amount']==0)].index, axis=0)
#3 rows dropped
train.shape
#Checking test data
#Again no records! AWESOME!
test.loc[((test['pickup_latitude']!=0) & (test['pickup_longitude']!=0))&((test['dropoff_latitude']==0) & (test['dropoff_longitude']==0))]
sns.distplot(train['H_Distance'])
high_distance = train.loc[(train['H_Distance']>100)&(train['fare_amount']!=0)]
high_distance
high_distance.shape
#high_distance['H_Distance'] = high_distance.apply(lambda row: (row['fare_amount'] - 2.50)/1.56,axis=1)
#The distance values have been replaced by the newly calculated ones according to the fare
high_distance.head()
#sync the train data with the newly computed distance values from high_distance dataframe
train.update(high_distance)
train.shape
train[train['H_Distance']==0]
train[(train['H_Distance']==0)&(train['fare_amount']==0)]
train = train.drop(train[(train['H_Distance']==0)&(train['fare_amount']==0)].index, axis = 0)
#4 rows dropped
train[(train['H_Distance']==0)].shape
#Between 6AM and 8PM on Mon-Fri
rush_hour = train.loc[(((train['Hour']>=6)&(train['Hour']<=20)) & ((train['Day of Week']>=1) & (train['Day of Week']<=5)) & (train['H_Distance']==0) & (train['fare_amount'] < 2.5))]
rush_hour
train=train.drop(rush_hour.index, axis=0)
train.shape
#Between 8PM and 6AM on Mon-Fri
non_rush_hour = train.loc[(((train['Hour']<6)|(train['Hour']>20)) & ((train['Day of Week']>=1)&(train['Day of Week']<=5)) & (train['H_Distance']==0) & (train['fare_amount'] < 3.0))]
#print(Counter(non_work_hours['Hour']))
#print(Counter(non_work_hours['Day of Week']))
non_rush_hour
#keep these. Since the fare_amount is not <2.5 (which is the base fare), these values seem legit to me.
#Saturday and Sunday all hours
weekends = train.loc[((train['Day of Week']==0) | (train['Day of Week']==6)) & (train['H_Distance']==0) & (train['fare_amount'] < 3.0)]
weekends
#Counter(weekends['Day of Week'])
#keep these too. Since the fare_amount is not <2.5, these values seem legit to me.
train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]
scenario_3 = train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]
len(scenario_3)
#We do not have any distance values that are outliers.
scenario_3.sort_values('H_Distance', ascending=False)
scenario_3['fare_amount']
train.update(scenario_3)
train.shape
train.loc[(train['H_Distance']==0) & (train['fare_amount']!=0)]
scenario_4 = train.loc[(train['H_Distance']==0) & (train['fare_amount']!=0)]
len(scenario_4)
#Using our prior knowledge about the base price during weekdays and weekends for the cabs.
#I do not want to impute these 1502 values as they are legible ones.
scenario_4.loc[(scenario_4['fare_amount']<=3.0)&(scenario_4['H_Distance']==0)]
scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]
scenario_4_sub = scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]
#scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]
len(scenario_4_sub)
scenario_4_sub['H_Distance'] = scenario_4_sub.apply(
lambda row: ((row['fare_amount']-2.50)/1.56), axis=1
)
train.update(scenario_4_sub)
train.shape
#train["hdist"] = train['H_Distance'] > 100
#train.drop("hdist", axis= 1, inplace =True)
train.head()
train.columns
test.columns
#not including the pickup_datetime columns as datetime columns cannot be directly used while modelling. Features need to extracted from the 
#timestamp fields which will later be used as features for modelling.
train = train.drop(['key','pickup_datetime'], axis = 1)
test = test.drop(['key','pickup_datetime'], axis = 1)
train.columns
test.columns
X = train.iloc[:,train.columns!='fare_amount']
Y = train['fare_amount'].values

X.shape, Y.shape
X.head()
X.columns
#x_train = x_train[[ 'H_Distance', 'pickup_longitude', 'Year', 'dropoff_longitude']]

#x_train = x_train[['H_Distance']]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 42)
scalar = StandardScaler().fit(X_train)
scalar.transform(X_train)

scalar.transform(test)

scalar.transform(X_test)
X_train.shape,Y_train.shape,X_test.shape,Y_test.shape
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)
lg_preds = lin_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(Y_test,lg_preds))
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train,Y_train)
tree_preds = tree_reg.predict(X_test)
np.sqrt(mean_squared_error(Y_test,tree_preds))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 5)
#@title Default title text
#reg.fit(X_train, Y_train)
rf.fit(X_train, Y_train)

rf_preds = rf.predict(X_test)
np.sqrt(mean_squared_error(Y_test,rf_preds))
from sklearn.model_selection import cross_val_score
cross_val_score(rf,X,Y,cv=5,scoring= "neg_mean_squared_error")
np.sqrt(15.3890688)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = [
    {'n_estimators': [3, 6, 10,20,30], 'max_features': [2, 4, 6, 8,10,12]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train,Y_train)
grid_preds = grid_search.predict(test)
grid_search.best_params_
grid_search.best_estimator_
#np.sqrt(mean_squared_error(Y_test,grid_preds))
#cvres = grid_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
     # print(np.sqrt(-mean_score), params)
submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = grid_preds
submission.to_csv('submission_1.csv', index=False)
submission.head(20)
import lightgbm as lgbm
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'verbose': 0,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_aplha': 1,
        'reg_lambda': 0.001,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1     
    }
pred_test_y = np.zeros(x_test.shape[0])
pred_test_y.shape
train_set = lgbm.Dataset(x_train, y_train, silent=True)
train_set
model = lgbm.train(params, train_set = train_set, num_boost_round=300)
print(model)
pred_test_y = model.predict(x_test, num_iteration = model.best_iteration)
print(pred_test_y)
submission['fare_amount'] = pred_test_y
submission.to_csv('submission_LGB.csv', index=False)
submission.head(20)
import xgboost as xgb 
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)
dtrain
#set parameters for xgboost
params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.05
         }
num_rounds = 50
xb = xgb.train(params, dtrain, num_rounds)
y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb)
submission['fare_amount'] = y_pred_xgb
submission.to_csv('submission_XGB.csv', index=False)
submission.head(20)
sns.heatmap(train.corr())



