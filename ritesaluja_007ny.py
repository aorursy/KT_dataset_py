# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import seaborn as sns
import matplotlib.pyplot as plt

import gc

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler
# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")
#the shortest distance over the earth’s surface
gc.collect()
R = 6371e3
a1 = np.radians(test['pickup_latitude'])
a2 = np.radians(test['dropoff_latitude'])
da = np.radians(test['dropoff_latitude']-test['pickup_latitude'])
dl = np.radians(test['dropoff_longitude']-test['pickup_longitude'])

test = test.drop(columns = ['pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude'])

a = np.sin(da/2) * np.sin(da/2) + np.cos(a1) * np.cos(a2) * np.sin(dl/2) * np.sin(dl/2)
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
d = R * c;

del R,c,a,a1,a2,da,dl

test['Distance']= pd.Series(d)
del d
gc.collect()
# Reading File
train_path  = '../input/train.csv'

# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())

dftemp = pd.read_csv(train_path, usecols=cols, dtype=traintypes, nrows = 500000)
#50000000
gc.collect()
# Save into feather format, about 1.5Gb. 
dftemp.to_feather('nyc_taxi_data_raw.feather')
del dftemp,traintypes,cols
gc.collect()
# load the same dataframe next time directly, without reading the csv file again!
df = pd.read_feather('nyc_taxi_data_raw.feather')

# It took less than one tenth of time to read the file
#the shortest distance over the earth’s surface
gc.collect()
R = 6371e3
a1 = np.radians(df['pickup_latitude'])
a2 = np.radians(df['dropoff_latitude'])
da = np.radians(df['dropoff_latitude']-df['pickup_latitude'])
dl = np.radians(df['dropoff_longitude']-df['pickup_longitude'])

df = df.drop(columns = ['pickup_latitude','dropoff_latitude','pickup_longitude','dropoff_longitude'])

a = np.sin(da/2) * np.sin(da/2) + np.cos(a1) * np.cos(a2) * np.sin(dl/2) * np.sin(dl/2)
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
d = R * c;

del R,c,a,a1,a2,da,dl

df['Distance']= pd.Series(d)
del d
gc.collect()
sns.lmplot(x='Distance',y='fare_amount', data = df)
#fig, (ax1,ax2) = plt.subplots(figsize=(12,9),ncols=2, nrows=1)
#sns.regplot(x='passenger_count'^2,y='fare_amount',data=df)
#gc.collect()
#sns.regplot(x='Distance',y='fare_amount',data=df)
#gc.collect()
df = df.dropna()
sns.distplot(df['passenger_count'])
gc.collect()
df = df[~(df[['passenger_count']] == 0).any(axis=1)]
scaler = MinMaxScaler()
gc.collect()
sns.set(style="whitegrid")
sns.boxplot(x="passenger_count", y="Distance", data=df)
plt.grid(True)
plt.show()
gc.collect()
#Linear regression, Lasso, LGB, XGBoost tried 

y = df['fare_amount']
#df['Distance_per_passenger'] = df['Distance']/df['passenger_count']
x = df.drop(columns=['pickup_datetime','fare_amount'])
#x = x.values
gc.collect()
scaler.fit_transform(x) 
gc.collect()
#regr = LinearRegression()

#svr_rbf = SVR(kernel='rbf', C=100, degree=3, gamma=1) # kernel coefficient
#svr_poly = SVR(kernel='poly', C=1, degree=2)
#svr_rbf.fit(x,y)
ada = RandomForestRegressor()
#xgb = XGBRegressor(eta=0.0001)
ada.fit(x,y)
y_pred = ada.predict(x)
print('The RMSE of prediction on training is:',mean_squared_error(y, y_pred) ** 0.5)
model = ada
"""
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20)
del x,y
gc.collect()
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
rfc = RandomForestRegressor() 
# Use a grid over parameters of interest
param_grid = { 
           "n_estimators" : [ 36, 45, 54, 63,100],
           "max_depth" : [ 5, 10, 15, 20,40],
            "min_samples_leaf" : [1,5,10,20,40]
           }
model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
model.fit(X_train, y_train)
print(model.best_params_)
y_pred = model.predict(X_val)
print('The rmse of prediction is:', mean_squared_error(y_val, y_pred) ** 0.5)
gc.collect()


X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20)
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)


# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 500,
    'feature_fraction': 0.9,
    'learning_rate': 0.005,
    'verbose': 0,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'early_stopping_round':100
}
#'bagging_fraction': 0.8,
#    'bagging_freq': 10,


print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                
                )
"""
gc.collect()
x_test = test.drop(columns=['key','pickup_datetime'])
gc.collect()
scaler.transform(x_test)
gc.collect()
y_test = model.predict(x_test)
y_test =  y_test
df = pd.DataFrame()
df["key"] = test["key"]
df["fare_amount"] = y_test
df.to_csv("sample_submission.csv", index = False)
