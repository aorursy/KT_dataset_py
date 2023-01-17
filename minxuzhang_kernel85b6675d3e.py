# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load some default Python modules
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
plt.style.use('seaborn-whitegrid')
#read data in 
df_test = pd.read_csv("../input/test.csv")
df_train =  pd.read_csv('../input/train.csv', nrows = 500_000, ) #1M to test models
df_train.dtypes
#Check Null Values
df_train.isnull().sum()
#Null dropoff_longitude and latitude is small, drop those data points
df_train=df_train.dropna()
df_test=df_test.dropna()
print(df_train.isnull().sum())
print(df_test.isnull().sum())
#Check Negative Fare
df_train.query("fare_amount <= 0").count()

df_test.dtypes
#Drop Negative data for train data, note that test data doesn't have `fare_amount` column
#Number of 0 or negative fare is small, drop the data points
df_train=df_train.query("fare_amount > 0")
#check
df_train.describe()
### Check to drop abnormally high fare
### fare distribution
fig=plt.figure(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
### fare distribution
sns.distplot(df_train.query('fare_amount<100')['fare_amount'])
#Zoom in for fare amount < 10
fig=plt.figure(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
### fare distribution
### zoom in, find min starting price
sns.distplot(df_train.query('fare_amount<10')['fare_amount'])
## previous graph shows min price is 2.5, delete all fare < 2.5 Note: df_test does not have fare_amount
df_train.query("fare_amount < 2.5").count()
#drop the data points
df_train=df_train.query("fare_amount >= 2.5")
#check
df_train.describe()
## check all fare > 100
df_train.query("fare_amount > 100").count()
df_train.query("fare_amount > 100").head(5)
## some data points has same pick up and drop off location, for example key 1335
#Check those with 0 longtidue and latitude
df_train.query('pickup_longitude *  pickup_latitude * dropoff_longitude * dropoff_latitude == 0').count() 
## df_train.query('(pickup_longitude == 0) or (pickup_latitude == 0)')
#Drop those with 0 longtitude or latitude for both train and test
df_train = df_train.query('pickup_longitude *  pickup_latitude * dropoff_longitude * dropoff_latitude > 0') 
#Check
print(df_train.describe())
print(df_test.describe())
## create a function to calculate distance
def haversine_distance(lat1, long1, lat2, long2):
    R = 6371  #radius of earth in kilometers
    #R = 3959 #radius of earth in miles
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    delta_phi = np.radians(lat2-lat1)
    delta_lambda = np.radians(long2-long1)

    #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    #c = 2 * atan2( √a, √(1−a) )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    #d = R*c
    d = (R * c) #in kilometers
    return d
## reset index since we dropped data
df_train = df_train.reset_index()
df_test=df_test.reset_index()
## Calculate distance for test set
for i in range(len(df_test)):    
    df_test.loc[[i], 'distance'] = haversine_distance(df_test.loc[i]['pickup_latitude'], df_test.loc[i]['pickup_longitude'], df_test.loc[i]['dropoff_latitude'], df_test.loc[i]['dropoff_longitude'])
#check
df_test.describe()
##Calculate distance for train dataset
for i in range(len(df_train)):    
    df_train.loc[[i], 'distance'] = haversine_distance(df_train.loc[i]['pickup_latitude'], df_train.loc[i]['pickup_longitude'], df_train.loc[i]['dropoff_latitude'], df_train.loc[i]['dropoff_longitude'])
df_train.describe()
### see how many data has 0 distance
df_train.query('distance == 0').count()
df_train['distance'] = pd.to_numeric(df_train['distance'])
df_train = df_train.query('distance > 0.0')
df_test['distance'] = pd.to_numeric(df_test['distance'])
###understand fare per mile distribution
df_train=df_train.assign(fare_per_mile = df_train.fare_amount/df_train.distance)
df_train.query('(fare_per_mile > 13) or (distance < 0.1)').count()
df_train = df_train.query('(fare_per_mile <= 13) and (distance > 0.1)')
df_train.describe()
#### next steps:
## 1. create new column for datetime
## 2. deal with lati, longti is 0
## 3. deal with very high fare
## 4. deal with passeger > 4
## 5. deal with longtitude, latitude
## 6. deal with 2.5 min fee
data = [df_train,df_test]
for i in data:
    i["pickup_datetime"] = pd.to_datetime(i["pickup_datetime"])
    i['year'] = i['pickup_datetime'].dt.year
    i['mth'] = i['pickup_datetime'].dt.month
    i['date'] = i['pickup_datetime'].dt.day
    i['day_of_week'] = i['pickup_datetime'].dt.dayofweek
    i['hour'] = i['pickup_datetime'].dt.hour
df_test.dtypes
X = df_train.drop(columns=['fare_amount', 'pickup_datetime', 'fare_per_mile', 'key'])
y = df_train['fare_amount'] - 2.5
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
# Create your housing DMatrix
## taxi_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.7],
    'n_estimators': [50],
    'max_depth': [5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error', cv=2, verbose=1)
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
#Read and preprocess test set
test_key = df_test['key']
x_pred = df_test.drop(columns=['key', 'pickup_datetime'])

#Predict from test set
prediction = grid_mse.predict(x_pred) + 2.5
#Create submission file
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction.round(2)
})

submission.to_csv('./taxi_fare_submission.csv',index=False)
submission.head()
