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
training_data=pd.read_csv('../input/train.csv',nrows=2000000)
test_data=pd.read_csv('../input/test.csv')
training_data
X_train=training_data.copy()
X_test=test_data.copy()
Y_train=training_data.copy()
X_train['pickup_datetime']=pd.to_datetime(X_train['pickup_datetime'])
X_train['hour']=X_train['pickup_datetime'].dt.hour
# X_train['dayofweek']=X_train['pickup_datetime'].dt.dayofweek
X_train['latitude_distance']=abs(X_train['dropoff_latitude']- X_train['pickup_latitude'])
X_train['longitude_distance']=abs(X_train['dropoff_longitude']- X_train['pickup_longitude'])
X_train=X_train.drop(columns=['dropoff_longitude','dropoff_latitude','key','fare_amount','pickup_datetime','passenger_count'])
X_test['pickup_datetime']=pd.to_datetime(X_test['pickup_datetime'])
X_test['hour']=X_test['pickup_datetime'].dt.hour
# X_test['dayofweek']=X_test['pickup_datetime'].dt.dayofweek
X_test['latitude_distance']=abs(X_test['dropoff_latitude']- X_test['pickup_latitude'])
X_test['longitude_distance']=abs(X_test['dropoff_longitude']- X_test['pickup_longitude'])
X_test=X_test.drop(columns=['dropoff_longitude','dropoff_latitude','key','pickup_datetime','passenger_count'])
Y_train=Y_train.drop(columns=['key','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count'])
"""
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout

model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

rmsprop=optimizers.RMSprop(lr=0.001)

model.compile(optimizer=rmsprop,loss='mse',metrics=['mae'])

model.fit(X_train,Y_train,epochs=4,batch_size=512)

Y_pred=model.predict(X_test)
"""
import xgboost as xgb

model=xgb.XGBRegressor()
model.fit(X_train, Y_train)

Y_pred=model.predict(data=X_test)
from matplotlib import pyplot as plt
from xgboost import plot_importance

plot_importance(model)
plt.show()
submission=test_data.copy()
submission=submission.drop(columns=['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count'])
submission['fare_amount']=Y_pred
submission.to_csv('sample_submission.csv',index=False)