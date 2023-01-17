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
training_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
training_data.head()
training_data['date']= pd.to_datetime(training_data['date'])
test_data['date']= pd.to_datetime(test_data['date'])
X_train=training_data.loc[:,['date','store','item']]
Y_train=training_data.loc[:,['sales']]
X_test=test_data.loc[:,['date','store','item']]
X_train['year']=X_train['date'].dt.year
X_train['month']=X_train['date'].dt.month
X_train['day']=X_train['date'].dt.day
X_train['weekday']=X_train['date'].dt.weekday
X_test['year']=X_test['date'].dt.year
X_test['month']=X_test['date'].dt.month
X_test['day']=X_test['date'].dt.day
X_test['weekday']=X_test['date'].dt.weekday
X_train= X_train.drop(columns=['date'])
X_test= X_test.drop(columns=['date'])
from keras import models
from keras import layers
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import History
from matplotlib import pyplot as plt

history=History()

model_DL=models.Sequential()
model_DL.add(layers.Dense(32, activation='relu',input_dim=6))
model_DL.add(Dropout(0.2))
model_DL.add(layers.Dense(32, activation='relu',input_dim=6))
model_DL.add(Dropout(0.2))

model_DL.add(layers.Dense(1,activation='softmax'))

adam=optimizers.Adagrad(lr=0.0001)

model_DL.compile(optimizer=adam,loss='mse')

history= model_DL.fit(X_train,Y_train,batch_size=512,epochs=4)

Y_pred_DL=model_DL.predict(X_test)
training_data.item.unique()
training_data.store.unique()
Y_pred_DL
from fbprophet import Prophet

submission=pd.DataFrame()
forecast_values=pd.Series([])
index=0
for item in training_data['item'].unique():
    for store in training_data['store'].unique():
        temp_training = training_data.loc[(training_data['store']==store) & (training_data['item']== item)]
        temp_training=temp_training[['date','sales']]
        temp_training.rename(columns={'date':'ds','sales':'y'}, inplace=True)
        model=Prophet()
        model.add_seasonality(name='monthly',period=30.5,fourier_order=5)
        model.fit(temp_training)
        future=model.make_future_dataframe(periods=90)
        forecast=model.predict(future)
        forecast=forecast.tail(90)
        forecast_values=forecast_values.append(forecast['yhat'],ignore_index=True)
        index=index+1
        print("Iteration",index)
submission['id']=test_data['id']
submission['sales']=forecast_values
submission
submission.to_csv('sample_submission.csv',index=False)