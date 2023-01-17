# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


%matplotlib inline

import json

import os

import pandas as pd

import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection  import train_test_split

import numpy as np

import gc

from scipy.stats import norm # for scientific Computing

from scipy import stats, integrate

import matplotlib.pyplot as plt


ASHRAE_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

ASHRAE_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

building_meta = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')


def reduce_memory_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
reduce_memory_usage(building_meta)

reduce_memory_usage(weather_train)

reduce_memory_usage(ASHRAE_train)



reduce_memory_usage(weather_test)

reduce_memory_usage(ASHRAE_test)
def plot_bar(data, name):

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111)

    data_label = data[name].value_counts()

    dict_train = dict(zip(data_label.keys(), ((data_label.sort_index())).tolist()))

    names = list(dict_train.keys())

    values = list(dict_train.values())

    plt.bar(names, values, color='green')

    ax.set_xticklabels(names, rotation=45)

    plt.grid()

    plt.show()

    

plot_bar(building_meta, 'primary_use')


BuildingTrain = building_meta.merge(ASHRAE_train, left_on='building_id', right_on='building_id' , how='left')

BuildingTest = building_meta.merge(ASHRAE_test, left_on='building_id', right_on='building_id' , how='left')

BuildingTrain.shape, BuildingTest.shape
del ASHRAE_test

del ASHRAE_train

del building_meta

gc.collect()
BTW_train=BuildingTrain.merge(weather_train,left_on=['site_id','timestamp'],right_on=['site_id','timestamp'],how='left')

BTW_test = BuildingTest.merge(weather_test,left_on=['site_id','timestamp'],right_on=['site_id','timestamp'],how='left')

BTW_train.shape
del BuildingTest

del BuildingTrain

del weather_test

del weather_train

gc.collect()
corrmat=BTW_train.corr()

plt.figure(figsize = (20,11))

sns.heatmap(corrmat,cmap=plt.cm.RdYlBu_r,vmin=-0.25,

            annot=True,vmax=0.6)
BTW_train = BTW_train.drop(columns=['year_built', 'floor_count', 'wind_direction', 'dew_temperature'])

BTW_test = BTW_test.drop(columns=['year_built', 'floor_count','wind_direction', 'dew_temperature'])
BTW_train ['timestamp'] =  pd.to_datetime(BTW_train['timestamp'])

BTW_test ['timestamp'] =  pd.to_datetime(BTW_test['timestamp'])

BTW_train['Month']=pd.DatetimeIndex(BTW_train['timestamp']).month

BTW_test['Month']=pd.DatetimeIndex(BTW_test['timestamp']).month

BTW_train['Day']=pd.DatetimeIndex(BTW_train['timestamp']).day

BTW_test['Day']=pd.DatetimeIndex(BTW_test['timestamp']).day


BTW_train= BTW_train.groupby(['meter',BTW_train['building_id'],'primary_use',BTW_train['Month'], BTW_train['Day']]).agg({'meter_reading':'sum', 'air_temperature': 'mean', 'wind_speed': 'mean', 'precip_depth_1_hr': 'mean', 'cloud_coverage': 'mean', 'square_feet': 'mean'})

BTW_test_1= BTW_test.groupby(['row_id','meter',BTW_test['building_id'],'primary_use',BTW_test['Month'], BTW_test['Day']]).agg({ 'air_temperature': 'mean', 'wind_speed': 'mean', 'precip_depth_1_hr': 'mean', 'cloud_coverage': 'mean', 'square_feet': 'mean'})
BTW_train = BTW_train.reset_index()
BTW_train['wind_speed'] = BTW_train['wind_speed'].astype('float32')

BTW_train['air_temperature'] = BTW_train['air_temperature'].astype('float32')

BTW_train['precip_depth_1_hr'] = BTW_train['precip_depth_1_hr'].astype('float32')

BTW_train['cloud_coverage'] = BTW_train['cloud_coverage'].astype('float32')

BTW_test['wind_speed'] = BTW_test['wind_speed'].astype('float32')

BTW_test['air_temperature'] = BTW_test['air_temperature'].astype('float32')

BTW_test['precip_depth_1_hr'] = BTW_test['precip_depth_1_hr'].astype('float32')

BTW_test['cloud_coverage'] = BTW_test['cloud_coverage'].astype('float32')
BTW_train['precip_depth_1_hr'].fillna(method='ffill', inplace = True)

BTW_train['cloud_coverage'].fillna(method='bfill', inplace = True)



BTW_train['wind_speed'].fillna(BTW_train['wind_speed'].mean(), inplace=True)

BTW_train['air_temperature'].fillna(BTW_train['air_temperature'].mean(), inplace=True)



BTW_test['precip_depth_1_hr'].fillna(method='ffill', inplace = True)

BTW_test['cloud_coverage'].fillna(method='bfill', inplace = True)

BTW_test['precip_depth_1_hr'].fillna(BTW_test['precip_depth_1_hr'].mean(), inplace=True)

BTW_test['cloud_coverage'].fillna(BTW_test['cloud_coverage'].mean(), inplace=True)



BTW_test['wind_speed'].fillna(BTW_test['wind_speed'].mean(), inplace=True)

BTW_test['air_temperature'].fillna(BTW_test['air_temperature'].mean(), inplace=True)

BTW_train.isnull().sum()
BTW_train.primary_use.unique()
BTW_encoded = BTW_train[:]

BTW_test_encoded = BTW_test[:]





from sklearn.preprocessing import LabelEncoder





le = LabelEncoder()



BTW_encoded["primary_use"] = le.fit_transform(BTW_encoded["primary_use"])

BTW_test_encoded["primary_use"] = le.fit_transform(BTW_test_encoded["primary_use"])
X = BTW_encoded[['meter', 'building_id', 'primary_use', 'Month', 'Day','air_temperature', 'wind_speed', 'precip_depth_1_hr', 'cloud_coverage',

       'square_feet']]

y = BTW_encoded['meter_reading']
x_train, x_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state= 45)


from sklearn import preprocessing

from keras import backend as K

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization

from keras.models import Sequential

from keras.optimizers import RMSprop,Adam

from keras import regularizers
def root_mean_squared_error(y_true, y_pred):

  return K.sqrt(K.mean(K.square(y_pred - y_true)))
def run_model(model,x_train,y_train,epochs=50,batch_size=500,verbose=1,validation_data=(x_val,y_val),callbacks =None):

  x_train = x_train.values[:]

  x_train= x_train.reshape((x_train.shape[0],1,x_train.shape[-1]))

  y_train = np.log1p(y_train)

  if validation_data != None:

    x_val = validation_data[0].values[:]

    x_val = x_val.reshape((x_val.shape[0],1,x_val.shape[-1]))

    y_val = np.log1p(validation_data[-1])

      

  return model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=verbose,validation_data=(x_val,y_val),callbacks=callbacks)


es = EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0.0001, patience=5, verbose=True, mode='auto')
from keras.layers import SimpleRNN
def make_model_rnn(input_dim=10,metrics=root_mean_squared_error,loss='mse', optimizer="rmsprop",drop_rate=0.2):



  model = Sequential()

  model.add(SimpleRNN(128,return_sequences=True, input_shape=(None,input_dim)))

  model.add(Dropout(drop_rate))

  model.add(BatchNormalization())

  model.add(SimpleRNN(128,return_sequences=False))

  model.add(BatchNormalization())

  model.add(Dropout(drop_rate))

  model.add(Dense(1))

  model.compile(optimizer=optimizer, loss=loss, metrics=[metrics, 'accuracy'])

  

  return model
simple_rnn_model = make_model_rnn(input_dim=x_train.shape[-1],drop_rate=0.2)
simple_rnn_model.summary()
history = run_model(simple_rnn_model,x_train,y_train,epochs=30,batch_size=500,verbose=1,validation_data=(x_val,y_val), callbacks =[es]) # callbacks =[mc, es]
loss = history.history

loss.keys()
rmse_loss_train = loss['root_mean_squared_error']

rmse_loss_val = loss['val_root_mean_squared_error']

epochs_stops = es.stopped_epoch +1 # epochs number from early stopping

epochs = range(1,epochs_stops + 1)  #len(loss_train)

plt.figure(figsize=(12,6))

plt.plot(epochs,rmse_loss_train,'r', label='RMSE train loss')

plt.plot(epochs,rmse_loss_val,'b',label='RMSE val loss')

plt.title(' root mean square error loss')

plt.legend()

plt.show()
def make_model(input_dim=10,metrics=root_mean_squared_error,loss='mse', optimizer="rmsprop",drop_rate=0.2):



  model = Sequential()

  model.add(GRU(128,return_sequences=True, input_shape=(None,input_dim)))

  model.add(Dropout(drop_rate))

  model.add(BatchNormalization())

  model.add(GRU(128,return_sequences=False))

  model.add(BatchNormalization())

  model.add(Dropout(drop_rate))

  model.add(Dense(1))

  model.compile(optimizer=optimizer, loss=loss, metrics=[metrics, 'accuracy'])

  

  return model
gru_rnn_model = make_model_rnn(input_dim=x_train.shape[-1],drop_rate=0.2)
gru_rnn_model.summary()
history = run_model(gru_rnn_model,x_train,y_train,epochs=30,batch_size=500,verbose=1,validation_data=(x_val,y_val), callbacks =[es]) # callbacks =[mc, es]G
loss = history.history

loss.keys()
rmse_loss_train = loss['root_mean_squared_error']

rmse_loss_val = loss['val_root_mean_squared_error']

epochs_stops = es.stopped_epoch +1 # epochs number from early stopping

epochs = range(1,epochs_stops + 1)  #len(loss_train)

plt.figure(figsize=(12,6))

plt.plot(epochs,rmse_loss_train,'r', label='RMSE train loss')

plt.plot(epochs,rmse_loss_val,'b',label='RMSE val loss')

plt.title(' root mean square error loss')

plt.legend()

plt.show()
def make_model(input_dim=10,metrics=root_mean_squared_error,loss='mse', optimizer="rmsprop",drop_rate=0.2):



  model = Sequential()

  model.add(LSTM(128,return_sequences=True, input_shape=(None,input_dim)))

  model.add(Dropout(drop_rate))

  model.add(BatchNormalization())

  model.add(LSTM(128,return_sequences=False))

  model.add(BatchNormalization())

  model.add(Dropout(drop_rate))

  model.add(Dense(1))

  model.compile(optimizer=optimizer, loss=loss, metrics=[metrics, 'accuracy'])

  

  return model
model = make_model(input_dim=x_train.shape[-1],drop_rate=0.2)
model.summary()
history = run_model(model,x_train,y_train,epochs=30,batch_size=500,verbose=1,validation_data=(x_val,y_val), callbacks =[es]) # callbacks =[mc, es]
loss = history.history

loss.keys()
rmse_loss_train = loss['root_mean_squared_error']

rmse_loss_val = loss['val_root_mean_squared_error']

epochs_stops = es.stopped_epoch +1 # epochs number from early stopping

epochs = range(1,epochs_stops + 1)  #len(loss_train)

plt.figure(figsize=(12,6))

plt.plot(epochs,rmse_loss_train,'r', label='RMSE train loss')

plt.plot(epochs,rmse_loss_val,'b',label='RMSE val loss')

plt.title(' root mean square error loss')

plt.legend()

plt.show()
# submit = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

# x_test = BTW_test[['meter', 'building_id', 'primary_use', 'Month', 'Day','air_temperature', 'wind_speed', 'precip_depth_1_hr', 'cloud_coverage',#

#        'square_feet']]

# x_test = x_test.values[:]

# x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[-1]))

# prediction = historyG.predict(x_test)

# # We proceed with expo function

# prediction = np.expm1(prediction)

# submit['meter_reading'] = prediction

# submit.to_csv('submission.csv', index=False,float_format='%.4f')