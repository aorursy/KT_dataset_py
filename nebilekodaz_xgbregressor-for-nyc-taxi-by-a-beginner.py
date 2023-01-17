# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_set = pd.read_csv('../input/train.csv', nrows = 10 ** 6)
test_set = pd.read_csv('../input/test.csv')

test_set.head()
import datetime
train_set.info()
#convert the pickup_datetime to datetime data type. It helps to extract new features.
train_set['pickup_datetime_dt']=pd.to_datetime(train_set.pickup_datetime)
test_set['pickup_datetime_dt']=pd.to_datetime(test_set.pickup_datetime)
train_set.info()
train_set['Year'] = train_set['pickup_datetime_dt'].dt.year
train_set['Month'] = train_set['pickup_datetime_dt'].dt.month
train_set['Day'] = train_set['pickup_datetime_dt'].dt.day
train_set['Hour'] = train_set['pickup_datetime_dt'].dt.hour
train_set['Dayoftheweek'] = train_set['pickup_datetime_dt'].dt.dayofweek
test_set['Year'] = test_set['pickup_datetime_dt'].dt.year
test_set['Month'] = test_set['pickup_datetime_dt'].dt.month
test_set['Day'] = test_set['pickup_datetime_dt'].dt.day
test_set['Hour'] = test_set['pickup_datetime_dt'].dt.hour
test_set['Dayoftheweek'] = test_set['pickup_datetime_dt'].dt.dayofweek
#drop nan values(missing values)
train_set = train_set.dropna(how = 'any', axis = 0)
#data visualization for outliers
plt.figure(figsize=(15,8))
sns.distplot(train_set['pickup_latitude'])
plt.figure(figsize=(15,8))
sns.distplot(train_set['pickup_longitude'])
plt.figure(figsize=(15,8))
sns.distplot(train_set['dropoff_longitude'])
plt.figure(figsize=(15,8))
sns.distplot(train_set['dropoff_latitude'])
#detecting outliers
def find_lower_outlier(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    lower = q1 - 1.5 * (q3 - q1)
    return lower
def find_upper_outlier(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    upper = q3 + 1.5 * (q3 - q1)
    return upper           
pi_long_lower=find_lower_outlier(train_set['pickup_longitude'])
pi_long_lower
pi_long_upper=find_upper_outlier(train_set['pickup_longitude'])
pi_long_upper
pi_lati_lower=find_lower_outlier(train_set['pickup_latitude'])
pi_lati_lower
pi_lati_upper=find_upper_outlier(train_set['pickup_latitude'])
pi_lati_upper
dro_long_lower=find_lower_outlier(train_set['dropoff_longitude'])
dro_long_lower
dro_long_upper=find_upper_outlier(train_set['dropoff_longitude'])
dro_long_upper
dro_lati_lower=find_lower_outlier(train_set['dropoff_latitude'])
dro_lati_lower
dro_lati_upper=find_upper_outlier(train_set['dropoff_latitude'])
dro_lati_upper
train_set['pickup_longitude']=train_set['pickup_longitude'].mask(train_set['pickup_longitude']<pi_long_lower,train_set['pickup_longitude'].mean())
train_set['pickup_longitude']=train_set['pickup_longitude'].mask(train_set['pickup_longitude']>pi_long_upper,train_set['pickup_longitude'].mean())
train_set['pickup_latitude']=train_set['pickup_latitude'].mask(train_set['pickup_latitude']<pi_long_lower,train_set['pickup_latitude'].mean())
train_set['pickup_latitude']=train_set['pickup_latitude'].mask(train_set['pickup_latitude']>pi_long_upper,train_set['pickup_latitude'].mean())
train_set['dropoff_longitude']=train_set['dropoff_longitude'].mask(train_set['dropoff_longitude']<pi_long_lower,train_set['dropoff_longitude'].mean())
train_set['dropoff_longitude']=train_set['dropoff_longitude'].mask(train_set['dropoff_longitude']>pi_long_upper,train_set['dropoff_longitude'].mean())
train_set['dropoff_latitude']=train_set['dropoff_latitude'].mask(train_set['dropoff_latitude']<pi_long_lower,train_set['dropoff_latitude'].mean())
train_set['dropoff_latitude']=train_set['dropoff_latitude'].mask(train_set['dropoff_latitude']>pi_long_upper,train_set['dropoff_latitude'].mean())
train_set.head()
cols=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','Year','Month','Day','Hour','Dayoftheweek']
col=['fare_amount']
X_train=train_set[cols]
y_train=train_set[col]
X_test=test_set[cols]

y_train.head()
#feature scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # create an instance
scaler.fit(X_train[cols])  
import xgboost as xgb
from sklearn.metrics import mean_squared_error
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train[cols], y_train, verbose=False) 
pred = xgb_model.predict(X_train[cols])
print('xgb train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = xgb_model.predict(X_test[cols])
submission = pd.DataFrame({
        "key": test_set['key'],
        "fare_amount": pred.round(2)
})
submission.to_csv('sub_fare.csv',index=False)
submission
