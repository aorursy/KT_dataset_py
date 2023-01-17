import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
train_df = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test_df = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
train_df.head()
train_df.isnull().sum()
train_df = train_df.fillna('')
train_df['Country/Province'] = train_df['Country_Region'] + '/' + train_df['Province_State']
train_df = train_df.drop(['Province_State','Country_Region','Id'],axis=1)
train_df[train_df['Country/Province']=='Afghanistan/'].count()
train_df['Country/Province'].nunique()
train_df['Country/Province']
22950/306
train_df.head()
train_df['Date']
from sklearn.preprocessing import MinMaxScaler
mms1 = MinMaxScaler((0,1))

mms2 = MinMaxScaler((0,1))
train_df['ConfirmedCases'] = mms1.fit_transform(train_df['ConfirmedCases'].values.reshape(-1, 1) )

train_df['Fatalities'] = mms2.fit_transform(train_df['Fatalities'].values.reshape(-1, 1) )
train_df['ConfirmedCases'].max()
confirmed_train_df = train_df.pivot_table(index='Country/Province', columns='Date',values='ConfirmedCases',fill_value=0).reset_index(drop=True)
confirmed_train_df
from tensorflow.keras.layers import Dense,Dropout,LSTM,Flatten,Dropout

from tensorflow.keras.models import Sequential

import tensorflow as tf
model=Sequential()
model.add(LSTM(250,activation='relu', input_shape=(75, 1)))

model.add(Dense(250))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),loss='mse')
X_train_confirmed = confirmed_train_df.iloc[:,:75].values

y_train_confirmed = confirmed_train_df.iloc[:,75].values

X_val_confirmed = confirmed_train_df.iloc[:20,:75].values

y_val_confirmed = confirmed_train_df.iloc[:20,75].values

X_train_confirmed = X_train_confirmed.reshape(306,75,1)

X_val_confirmed = X_val_confirmed.reshape(20,75,1)
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(verbose=1,patience=10)
model.fit(X_train_confirmed,y_train_confirmed,epochs=200,shuffle=False,callbacks=[es],validation_data=(X_val_confirmed,y_val_confirmed))
pd.DataFrame(model.history.history).plot()
n_features = 306

n_input = 75

forecast = []

batch=confirmed_train_df.iloc[:,-n_input:].values

current_batch=batch.reshape(306,75,1)

for i in range(43):

  current_pred=model.predict(current_batch)

  forecast.append(current_pred)

  current_batch=np.append(current_batch[:,1:,:],current_pred.reshape(306,1,1),axis=1)
pred_list_confirmed = np.array(forecast)
fatalities_train_df = train_df.pivot_table(index='Country/Province', columns='Date',values='Fatalities',fill_value=0).reset_index(drop=True)
fatalities_train_df
model=Sequential()

model.add(LSTM(250,activation='relu', input_shape=(75, 1)))

model.add(Dense(250))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),loss='mse')
X_train_fatalities = fatalities_train_df.iloc[:,:75].values

y_train_fatalities = fatalities_train_df.iloc[:,75].values

X_val_fatalities = fatalities_train_df.iloc[:20,:75].values

y_val_fatalities = fatalities_train_df.iloc[:20,75].values

X_train_fatalities = X_train_fatalities.reshape(306,75,1)

X_val_fatalities = X_val_fatalities.reshape(20,75,1)
model.fit(X_train_fatalities,y_train_fatalities,epochs=200,shuffle=False,callbacks=[es],validation_data=(X_val_fatalities,y_val_fatalities))
pd.DataFrame(model.history.history).plot()
n_features = 306

n_input = 75

forecast = []

batch=fatalities_train_df.iloc[:,-n_input:].values

current_batch=batch.reshape(306,75,1)

for i in range(43):

  current_pred=model.predict(current_batch)

  forecast.append(current_pred)

  current_batch=np.append(current_batch[:,1:,:],current_pred.reshape(306,1,1),axis=1)
pred_list_fatalities = np.array(forecast)
pred_list_fatalities.shape
pred_list_confirmed = pred_list_confirmed.reshape(43,306)
pred_list_fatalities = pred_list_fatalities.reshape(43,306)
pred_list_confirmed = pred_list_confirmed.transpose().reshape(13158,)

pred_list_fatalities = pred_list_fatalities.transpose().reshape(13158,)
pred_list_confirmed = mms1.inverse_transform(pred_list_confirmed.reshape(-1,1))

pred_list_fatalities = mms2.inverse_transform(pred_list_fatalities.reshape(-1,1))
pred_list_confirmed = pred_list_confirmed.round()

pred_list_fatalities = pred_list_fatalities.round()
test_df.head()
test_df['ConfirmedCases'] = pred_list_confirmed

test_df['Fatalities'] = pred_list_fatalities
test_df.drop(['Province_State','Country_Region','Date'],axis=1,inplace=True)
test_df.head()
test_df.to_csv('submission.csv',index=False)
test_df.max()
test_df.min()