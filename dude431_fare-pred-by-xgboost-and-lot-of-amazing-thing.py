import numpy as np
import pandas as pd 
import math
import os

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("../input/train.csv",nrows = (10 ** 7))
print("shape of train data", data.shape)

initial = len(data)
data.sample(10)
data.describe()
data.isnull().sum()
data = data[data.fare_amount >0] ## negative ones
data = data.dropna(how='any', axis=0) ## Null values

print(len(data))
plt.figure(figsize= (12,4))
sns.distplot(data['passenger_count'],hist=False )
plt.show()
#data[data.passenger_count>0].passenger_count.hist(bins=10, figsize = (16,8))
#plt.xlabel("Passanger Count")
#plt.ylabel("Frequency")
#plt.show()
print(len(data[data.passenger_count >6]))
data = data.drop(index= data[data.passenger_count >= 7].index, axis=0)
print(len(data))
plt.figure(figsize= (16,8))
sns.boxplot(x = data[data.passenger_count<7].passenger_count, y = data.fare_amount)
plt.show()
corr = data.corr()
plt.figure(figsize=(8,6))
sns.heatmap(data=corr, square=True , annot=True, cbar=True)
plt.show()
#our test data

test = pd.read_csv("../input/test.csv")
print("shape of test data", test.shape)
test.sample(5)
test.isnull().sum()
#it seems there in no null values in test dataset
test.describe()
print("_____Longitudes_____")
print(min(test.pickup_longitude.min(),test.dropoff_longitude.min()), \
max(test.pickup_longitude.max(),test.dropoff_longitude.max()))
print("\n")


print("_____Latitudes_____")
print(min(test.pickup_latitude.min(),test.dropoff_latitude.min()), \
max(test.pickup_latitude.max(),test.dropoff_latitude.max()))

data = data[(data.pickup_longitude >= -74.27) & (data.pickup_longitude <= -72.98) & \
        (data.pickup_latitude >= 40.56) & (data.pickup_latitude <= 41.71) & \
        (data.dropoff_longitude >= -74.27) & (data.dropoff_longitude <= -72.98) & \
        (data.dropoff_latitude >= 40.56) & (data.dropoff_latitude <= 41.71)]
print(len(data))
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))   # 2*R*asin...
data['distance_miles'] = distance(data.pickup_latitude, data.pickup_longitude, \
                                      data.dropoff_latitude, data.dropoff_longitude)
test['distance_miles'] = distance(test.pickup_latitude, test.pickup_longitude, \
                                      test.dropoff_latitude, test.dropoff_longitude)

print(len(data[data['distance_miles']==0]))

data = data.drop(index= data[(data['distance_miles']==0)].index, axis=0)
print(len(data))
plt.figure(figsize=(12,6))
sns.distplot( data["fare_amount"]<2.5,hist = False )
plt.show()
# We can drop such data where fare is less than 2.5 dollar
data[data['fare_amount'] < 2.5].shape

data = data.drop(index= data[(data['fare_amount']<2.5)].index, axis=0)
print(len(data))
fare_50 = data[(data['distance_miles']<1)&(data['fare_amount']>50)]
print(len(fare_50))
data = data.drop(index= data[(data['distance_miles']<1)&(data['fare_amount']>50)].index, axis=0)
print(len(data))
#here it is description of our final data
data.describe()
test_copy = test.copy()

train = data.drop(columns= ['key','pickup_datetime'], axis= 1).copy()
test = test.drop(columns= ['key','pickup_datetime'], axis= 1).copy()

print(train.shape)
print(test.shape)
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

stdscaler = StandardScaler()
xgb_train = stdscaler.fit_transform(train.drop('fare_amount', axis=1))
xgb_test =  stdscaler.fit_transform(test)
                       
x_train, x_test, y_train, y_test = train_test_split(xgb_train,train['fare_amount'], test_size=0.2, random_state = 42)
import xgboost as xgb

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'},
                    dtrain=matrix_train,
                    num_boost_round=100, 
                    early_stopping_rounds=10,
                    evals=[(matrix_test,'test')])
    return model

xgb_model = XGBmodel(x_train,x_test,y_train,y_test)
xgb_pred = xgb_model.predict(xgb.DMatrix(xgb_test), ntree_limit = xgb_model.best_ntree_limit)
test_copy['pred_xgb'] = xgb_pred

submission = pd.DataFrame(
    {'key': test_copy.key, 'fare_amount': test_copy.pred_xgb},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission_xgb.csv', index = False)

print(os.listdir('.'))