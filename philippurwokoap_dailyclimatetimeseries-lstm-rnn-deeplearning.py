# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

import datetime



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv')

test_data = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTest.csv')
train_data['date'] = pd.to_datetime(train_data['date'])

test_data['date'] = pd.to_datetime(test_data['date'])
def Visualize(kind='hist',figsize=(10,10)):

    def Col():

        cols = ['meantemp','humidity','wind_speed','meanpressure']

        for c in cols:

            yield c



    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=figsize)

    col = Col()

    for i in range(2):

        for j in range(2):

            curr = next(col)

            if kind == 'hist':

                axes[i,j].hist(train_data[curr])

            elif kind == 'plot':

                axes[i,j].plot(train_data['date'],train_data[curr])

                plt.gcf().autofmt_xdate()

            axes[i,j].set_title(curr)

    if kind=='hist':

        plt.savefig('data_hist.png')

    else:

        plt.savefig('data_plot.png')
Visualize('hist')
Visualize('plot',(20,10))
def process_data(data):

    x,y = [],[]

    switch = False

    

    if len(data)%2 == 1:

        last = False

    else:

        last = True

    

    for i in range(len(data)):

        if i == len(data)-1:

            if not last:

                break

        if switch:

            y.append(data[i])

            switch = False

        else:

            x.append(data[i])

            switch = True

    

    def reshape(d):

        d = np.array(d)

        d = np.reshape(d,(d.shape[0],1,1))

        return d

    return (reshape(x),np.array(y))
x_train_meantemp,y_train_meantemp = process_data(train_data.meantemp)

x_test_meantemp,y_test_meantemp = process_data(test_data.meantemp)
x_train_humidity,y_train_humidity = process_data(train_data.humidity)

x_test_humidity,y_test_humidity = process_data(test_data.humidity)
x_train_wind_speed,y_train_wind_speed = process_data(train_data.wind_speed)

x_test_wind_speed,y_test_wind_speed = process_data(test_data.wind_speed)
x_train_meanpressure,y_train_meanpressure = process_data(train_data.meanpressure)

x_test_meanpressure,y_test_meanpressure = process_data(test_data.meanpressure)
model_meantemp = keras.Sequential([

    keras.layers.LSTM(8,input_shape=(1,1,)),

    keras.layers.Dense(16,activation='relu'),

    keras.layers.Dense(32,activation='relu'),

    keras.layers.Dense(1)

])
model_humidity = keras.Sequential([

    keras.layers.LSTM(8,input_shape=(1,1,)),

    keras.layers.Dense(16,activation='relu'),

    keras.layers.Dense(32,activation='relu'),

    keras.layers.Dense(1)

])
model_wind_speed = keras.Sequential([

    keras.layers.LSTM(8,input_shape=(1,1,)),

    keras.layers.Dense(16,activation='relu'),

    keras.layers.Dense(32,activation='relu'),

    keras.layers.Dense(64,activation='relu'),

    keras.layers.Dense(1)

])
model_meanpressure = keras.Sequential([

    keras.layers.LSTM(8,input_shape=(1,1,)),

    keras.layers.Dense(16,activation='tanh'),

    keras.layers.Dense(32,activation='tanh'),

    keras.layers.Dense(64,activation='tanh'),

    keras.layers.Dense(1)

])
model_meantemp.compile(loss='mse',optimizer='adam')
model_humidity.compile(loss='mse',optimizer='adam')
model_wind_speed.compile(loss='mse',optimizer='adam')
model_meanpressure.compile(loss='mse',optimizer='adam')
model_meantemp.summary()
os.makedirs('logs',exist_ok=True)

logdir = os.path.join('logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
callback = keras.callbacks.TensorBoard(logdir)

earlyStoping = keras.callbacks.EarlyStopping(monitor='loss',patience=3)
history_humidity = model_humidity.fit(x_train_humidity,y_train_humidity,epochs=100,verbose=2,batch_size=16,callbacks=[callback,earlyStoping])
history_meantemp = model_meantemp.fit(x_train_meantemp,y_train_meantemp,epochs=100,verbose=2,batch_size=16,callbacks=[callback,earlyStoping])
history_wind_speed = model_wind_speed.fit(x_train_wind_speed,y_train_wind_speed,epochs=100,verbose=2,batch_size=16,callbacks=[callback,earlyStoping])
history_meanpressure = model_meanpressure.fit(x_train_meanpressure,y_train_meanpressure,epochs=1000,verbose=2,batch_size=16,callbacks=[callback,earlyStoping])
def Gen_hist():

    all_hist = ['history_humidity','history_meanpressure','history_meantemp','history_wind_speed']

    for hist in all_hist:

        yield hist
gen_hist = Gen_hist()

fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(10,10))

for i in range(2):

    for j in range(2):

        hist_now = next(gen_hist)

        axes[i,j].plot(eval(hist_now).history['loss'])

        axes[i,j].set_title(hist_now)

plt.savefig('loss_history.png')
def Gen_test():

    all_test = ['x_test_wind_speed','x_test_humidity','x_test_meantemp','x_test_meanpressure']

    all_y = ['y_test_wind_speed','y_test_humidity','y_test_meantemp','y_test_meanpressure']

    all_model = ['model_wind_speed','model_humidity','model_meantemp','model_meanpressure']

    for test in zip(all_test,all_y,all_model):

        yield test
gen_test = Gen_test()

fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(20,10))

for i in range(2):

    for j in range(2):

        test_now = next(gen_test)

        axes[i,j].plot(eval(test_now[2]).predict(eval(test_now[0])),label='Prediction')

        axes[i,j].plot(eval(test_now[1]),label='Actual')

        axes[i,j].set_title(test_now[0])

        axes[i,j].legend()

plt.savefig('prediction.png')
os.makedirs('models',exist_ok=True)
all_model = ['model_wind_speed','model_humidity','model_meantemp','model_meanpressure']

for model in all_model:

    eval(model).save(f'models/{model}.h5')