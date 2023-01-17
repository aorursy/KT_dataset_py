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
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv',nrows=10_000_000)
train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -150) & (train['pickup_longitude'] < 0)]
train = train.loc[(train['pickup_latitude'] > 0) & (train['pickup_latitude'] < 80)]
train = train.loc[(train['dropoff_longitude'] > -150) & (train['dropoff_longitude'] < 0)]
train = train.loc[(train['dropoff_latitude'] > 0) & (train['dropoff_longitude'] < 80)]
#train = train.loc[train[columns_to_select] < ]
# Let's assume taxa's can be mini-busses as well, so we select up to 8 passengers.
train = train.loc[train['passenger_count'] <= 8]
train.head()
test = pd.read_csv('../input/test.csv')
test.head()
train.dtypes
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r
def add_travel_distance_vector_features(df):
    df['distance'] = haversine(df.dropoff_longitude,df.dropoff_latitude,df.pickup_longitude,df.pickup_latitude)
add_travel_distance_vector_features(train)
add_travel_distance_vector_features(test)
train.head()
# train.drop(['dropoff_longitude','dropoff_latitude','pickup_longitude','pickup_latitude','pickup_datetime'],axis=1,inplace=True)
# test.drop(['dropoff_longitude','dropoff_latitude','pickup_longitude','pickup_latitude','pickup_datetime'],axis=1,inplace=True)
# train.drop(['pickup_datetime'],axis=1,inplace=True)
train.isnull().sum()
train.dropna(how = 'any', axis = 'rows', inplace=True)
train.describe().astype('float16')
sns.kdeplot(train.fare_amount, shade=True)
# train = train.loc[train['fare_amount'] <= 100]
# sns.kdeplot(train.fare_amount, shade=True)
train['key'] = pd.to_datetime(train.key)
test['key'] = pd.to_datetime(test.key)
train['hour'] = train['key'].dt.hour
test['hour'] = test['key'].dt.hour
train['day'] = train['key'].dt.day
test['day'] = test['key'].dt.day
train['month'] = train['key'].dt.month
test['month'] = test['key'].dt.month
train['year'] = train['key'].dt.year
test['year'] = test['key'].dt.year
train['daysinmonth'] = train['key'].dt.daysinmonth
test['daysinmonth'] = test['key'].dt.daysinmonth
train['weekofyear'] = train['key'].dt.weekofyear
test['weekofyear'] = test['key'].dt.weekofyear
train['dayofweek'] = train['key'].dt.dayofweek
test['dayofweek'] = test['key'].dt.dayofweek
train['dayofyear'] = train['key'].dt.dayofyear
test['dayofyear'] = test['key'].dt.dayofyear
train['quarter'] = train['key'].dt.quarter
test['quarter'] = test['key'].dt.quarter

train.key = train.key.values.astype(np.int64)
test.key = test.key.values.astype(np.int64)
train.pop('pickup_datetime')
test.pop('pickup_datetime')
train.pop('key')
test.pop('key')
train['longitude_distance'] = abs(train['pickup_longitude'] - train['dropoff_longitude'])
train['latitude_distance'] = abs(train['pickup_latitude'] - train['dropoff_latitude'])
test['longitude_distance'] = abs(test['pickup_longitude'] - test['dropoff_longitude'])
test['latitude_distance'] = abs(test['pickup_latitude'] - test['dropoff_latitude'])

train['distance_travelled_sin'] = np.sin((train['longitude_distance'] ** 2 * train['latitude_distance'] ** 2) ** .5)
test['distance_travelled_sin'] = np.sin((test['longitude_distance'] ** 2 * test['latitude_distance'] ** 2) ** .5)
train.head()
train.distance.describe().astype('float16')
y = train.pop('fare_amount')
X = train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=124)
from keras.layers import Dense, Input,Dropout,LeakyReLU,LSTM,BatchNormalization
from keras.models import Model
from keras import backend as Backend
import keras.activations as act
def rmse(y_true, y_pred):
    return Backend.sqrt(Backend.mean(Backend.square(y_pred - y_true), axis=-1))
# def nn(n_feature,k=1200):
#     model_in = Input(shape=(n_feature,))
#     model = BatchNormalization()(model_in)
#     model = Dense(k)(model)
#     model = LeakyReLU(0.15)(model)
#     model = Dropout(0.2)(model)
    
#     model = Dense(k)(model)
#     model = LeakyReLU(0.15)(model)
#     model = Dropout(0.2)(model)
    
#     model = Dense(k)(model)
#     model = LeakyReLU(0.15)(model)
#     model = Dropout(0.2)(model)
    
#     model = Dense(k)(model)
#     model = LeakyReLU(0.15)(model)
#     model = Dropout(0.2)(model)
    
#     model = Dense(1,activation=act.selu)(model)
    
#     model = Model(inputs=model_in,outputs=model)
#     model.compile(loss='mse',optimizer='nadam',metrics=[rmse])
#     return model
X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
test = np.reshape(test,(test.shape[0],1,test.shape[1]))
def lstm(n):
    model_in = Input(shape=(1,n))
#     model = BatchNormalization()(model_in)
    model = LSTM(100)(model_in)
    model = Dropout(0.2)(model)
#     model = LSTM(256,return_sequences=False)(model)
#     model = Dropout(0.2)(model)
    model = Dense(1,activation=act.selu)(model)
    model = Model(model_in,model)
    model.compile(loss='mse', optimizer='nadam',metrics=[rmse])
    return model
# model = nn(X.shape[1])
model = lstm(X_train.shape[2])
# history = model.fit(X_train,y_train,batch_size=1000,epochs=10,verbose=1)
history = model.fit(X_train,y_train,batch_size=10000,epochs=100,verbose=1,validation_data=(X_test,y_test))
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('model rmse')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.evaluate(X_test,y_test)
pres = model.predict(test)
test = pd.read_csv('../input/test.csv')
submission = pd.DataFrame(
    {'key': test.key, 'fare_amount': pres.reshape(9914)},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)
print(os.listdir('.'))
