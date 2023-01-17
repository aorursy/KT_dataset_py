# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read in 'TRAIN_DATA' values
TRAIN_DATA = 6000000
train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = TRAIN_DATA)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
train.head()
train.shape
test.shape
train.info()
# Checking the missing values
train.isnull().sum()
test.isnull().sum()
train.describe()
# Checking the rows with missing values
train[train.isnull().any(axis=1)]
#Drop the rows with missing values
train.dropna(inplace=True)
train.shape
train.columns
print('Pick-up boundaries of training set')
print('max value of longitude:',train['pickup_longitude'].max())
print('min value of longitude:',train['pickup_longitude'].min())
print('max value of latitude:',train['pickup_latitude'].max())
print('max value of latitude:',train['pickup_longitude'].max())
print("\n*******\n")
print('Drop-off boundaries of training set')
print('max value of longitude:',train['dropoff_longitude'].max())
print('min value of longitude:',train['dropoff_longitude'].min())
print('max value of latitude:',train['dropoff_latitude'].max())
print('max value of latitude:',train['dropoff_longitude'].max())
print('Pick-up boundaries of test set')
print('max value of longitude:',test['pickup_longitude'].max())
print('min value of longitude:',test['pickup_longitude'].min())
print('max value of latitude:',test['pickup_latitude'].max())
print('max value of latitude:',test['pickup_longitude'].max())
print("\n*******\n")
print('Drop-off boundaries of test set')
print('max value of longitude:',test['dropoff_longitude'].max())
print('min value of longitude:',test['dropoff_longitude'].min())
print('max value of latitude:',test['dropoff_latitude'].max())
print('max value of latitude:',test['dropoff_longitude'].max())
train = train[(-76 <= train['pickup_longitude']) & (train['pickup_longitude'] <= -72)]
train = train[(-76 <= train['dropoff_longitude']) & (train['dropoff_longitude'] <= -72)]
train = train[(38 <= train['pickup_latitude']) & (train['pickup_latitude'] <= 42)]
train = train[(38 <= train['dropoff_latitude']) & (train['dropoff_latitude'] <= 42)]
train.shape
# Checking the fare range
print("Max fare value:", train['fare_amount'].max())
print("Min fare value:", train['fare_amount'].min())
len(train[train['fare_amount']<0])
# Dropping the rows with fare value < 0
train = train[train['fare_amount']>=0]
train.shape
# Checking outliers in fare_amount
plt.figure(figsize=(12,4))
sns.boxplot(train['fare_amount'])
len(train[train['fare_amount']>200])
# Dropping the rows with fare value >200
train = train[train['fare_amount']<=200]
train.shape
import datetime
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime']) - datetime.timedelta(hours=4)
train['Year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Day'] = train['pickup_datetime'].dt.day
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minutes'] = train['pickup_datetime'].dt.minute
train['Day of Week'] = train['pickup_datetime'].dt.dayofweek
train.head(2)
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime']) - datetime.timedelta(hours=4)
test['Year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Day'] = test['pickup_datetime'].dt.day
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minutes'] = test['pickup_datetime'].dt.minute
test['Day of Week'] = test['pickup_datetime'].dt.dayofweek
from sklearn.metrics.pairwise import haversine_distances
from math import radians
def haversine(df):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1= np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    #### Based on the formula  x1=drop_lat,x2=dropoff_long 
    dlat = np.radians(df['dropoff_latitude']-df["pickup_latitude"])
    dlong = np.radians(df["dropoff_longitude"]-df["pickup_longitude"])
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong/2)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = 3956 #  Radius of earth in miles. Use 6371 for kilometers
    return c * r
train['Total distance']=haversine(train)

test['Total distance']=haversine(test)
test.head(2)
# Fare vs distance (Taking 1000 samples)
sns.lmplot(x='Total distance', y='fare_amount', data=train[1:1000])
def is_peak_hour(df):
    peak = False
    if df['Day of Week'] >= 0 and df['Day of Week'] <= 4:
        if (df['Hour'] >= 8 and df['Hour'] <= 10) or (df['Hour'] >= 16 and df['Hour'] <= 18):
            peak = True
        else:
            peak = False
    else:
        peak = False
    return peak
train['Peak hours'] = train.apply(is_peak_hour, axis=1)
test['Peak hours'] = test.apply(is_peak_hour, axis=1)
early_late_hours = [0,1,2,3,4,5,22,23]
train['Early late hours'] = train['Hour'].apply(lambda x: x in early_late_hours)
test['Early late hours'] = test['Hour'].apply(lambda x: x in early_late_hours)
train.head()
test.head(2)
# Just to make sure, everything will be in numbers
train['Peak hours'] = train['Peak hours'].replace({True: 1, False: 0})
train['Early late hours'] = train['Early late hours'].replace({True: 1, False: 0})
test['Peak hours'] = test['Peak hours'].replace({True: 1, False: 0})
test['Early late hours'] = test['Early late hours'].replace({True: 1, False: 0})
test.head(2)
train.head()
drop_columns = ['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude', 'Minutes']
train.head(2)
train.drop(drop_columns, axis = 1, inplace=True)
test.drop(drop_columns, axis = 1, inplace=True)
# Trying by dropping more columns
#drop_columns_addtnl = ['Day', 'Hour', 'Day of Week']
#train.drop(drop_columns_addtnl, axis = 1, inplace=True)
#test.drop(drop_columns_addtnl, axis = 1, inplace=True)

train.head()
test.head(2)
from sklearn.model_selection import train_test_split
X = train.drop(['fare_amount'], axis=1).values
y = train['fare_amount'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
import xgboost
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
xgb_pred = xgb_reg.predict(X_val)
from sklearn.metrics import mean_absolute_error, mean_squared_error
xgb_mae = mean_absolute_error(y_val, xgb_pred)
xgb_mse = mean_squared_error(y_val, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
print('XG Boost Regressor Performance:-')
print(f'MAE: {xgb_mae}\nMSE:{xgb_mse}\nRMSE:{xgb_rmse}')
sns.distplot(y_val-xgb_pred)
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
# Scale data
# Note: Scaling is needed for DL models
scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test)
NN_model = Sequential()

# Input Layer
NN_model.add(Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]))

# Hidden Layers
NN_model.add(Dense(256, activation='relu'))
NN_model.add(Dense(256, activation='relu'))
NN_model.add(Dense(256, activation='relu'))

# Output Layer
NN_model.add(Dense(1, activation='linear'))
#Compiling the model
NN_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
NN_model.summary()
# Fitting the model
NN_history = NN_model.fit(x=X_train_scaled, y=y_train, batch_size=512, epochs=50, 
                    validation_data=(X_val_scaled, y_val), shuffle=True)
from keras.layers import Dropout, BatchNormalization
NN_model1 = Sequential()

# Input Layer
NN_model1.add(Dense(128, kernel_initializer='normal', activation='relu', input_dim=X_train_scaled.shape[1]))
#NN_model1.add(BatchNormalization())

# Hidden Layers
NN_model1.add(Dense(256, kernel_initializer='normal', activation='relu'))

NN_model1.add(Dense(256, kernel_initializer='normal', activation='relu'))

NN_model1.add(Dense(256, kernel_initializer='normal', activation='relu'))

NN_model1.add(Dense(256, kernel_initializer='normal', activation='relu'))

NN_model1.add(Dense(256, kernel_initializer='normal', activation='relu'))

# Output Layer
NN_model1.add(Dense(1, kernel_initializer='normal', activation='linear'))

#Compiling the model
NN_model1.compile(loss='mse', optimizer='adam', metrics=['mae'])
NN_model1.summary()
# Fitting the model
NN_history1 = NN_model1.fit(x=X_train_scaled, y=y_train, batch_size=512, epochs=50, 
                    validation_data=(X_val_scaled, y_val), shuffle=True)
# Plot the loss of NN_history
plt.plot(NN_history.history['loss'], label='train loss')
plt.plot(NN_history.history['val_loss'], label='valdn loss')
plt.legend()
plt.show()
print('Model:NN_history')
print('Min value of training Loss:', min(NN_history.history['loss']))
print('Min value of validation Loss:', min(NN_history.history['val_loss']))
# Plot the loss of NN_history1
plt.plot(NN_history1.history['loss'], label='train loss')
plt.plot(NN_history1.history['val_loss'], label='valdn loss')
plt.legend()
plt.show()
print('Model:NN_history1')
print('Min value of training Loss:', min(NN_history1.history['loss']))
print('Min value of validation Loss:', min(NN_history1.history['val_loss']))
# Make prediction with NN_model1
NN_prediction = NN_model1.predict(test_scaled, verbose=1)
NN_prediction
submission = pd.read_csv('../input/new-york-city-taxi-fare-prediction/sample_submission.csv')
submission['fare_amount'] = NN_prediction
submission.to_csv('submission_NN.csv', index=False)
submission.head()
print('Saved file: ' + filename)