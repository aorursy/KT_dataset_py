import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow

from tensorflow import keras

import datetime

import os
dataframe = pd.read_csv('../input/population-time-series-data/POP.csv')

dataframe = dataframe[['date','value']]
dataframe['date'] = pd.to_datetime(dataframe['date'])
dataframe.tail()
plt.plot(dataframe.date,dataframe.value)

plt.title('Population 1950-2019')

plt.savefig('Population-Past-Now.png')
def split_sequence(sequence, n_steps=3):

    sequence = list(sequence)

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    def reshape(d):

        d = np.array(d)

        d = np.reshape(d,(d.shape[0],d.shape[1],1))

        return d

    return reshape(X), np.array(y)
train_data = dataframe.value.iloc[:700]

test_data = dataframe.value.iloc[700:]
x_train,y_train = split_sequence(train_data)

x_test,y_test = split_sequence(test_data)
model = keras.Sequential([

    keras.layers.LSTM(64,input_shape=(3,1,),activation='relu',return_sequences=True),

    keras.layers.LSTM(64,activation='relu'),

    keras.layers.Dense(1)

])
model.compile(loss='mse',optimizer='adam')
model.summary()
%load_ext tensorboard
os.makedirs('logs',exist_ok=True)

logdir = os.path.join('logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
callback = keras.callbacks.TensorBoard(logdir)

earlyStoping = keras.callbacks.EarlyStopping(monitor='loss',patience=3)
history = model.fit(x_train,y_train,epochs=100,batch_size=32,callbacks=[callback,earlyStoping],verbose=2)
plt.plot(history.history['loss'])

plt.title('RNN Model Training Loss')

plt.savefig('RNNModel-TrainingLoss.png')
plt.plot(model.predict(x_test),label='Prediction')

plt.plot(y_test,label='Actual')

plt.legend()

plt.title('Prediction Demonstration (Test)')

plt.savefig('PredictionDemonstration-Test.png')
def predict_future(shift_count):

    def reshape(three):

        return np.array(three).reshape(1,3,1) 

    array =  list(dataframe.value) + []

    now = len(dataframe)-3

    last = len(dataframe)

    for _ in range(shift_count):

        converted = reshape(array[now:last])

        array.append(model.predict(converted)[0][0])

        now += 1

        last += 1

    return array
future_prediction = predict_future(1000)
plt.figure(figsize=(10,5))

plt.plot(future_prediction,'--',label='Prediction')

plt.plot(dataframe.value,label='Actual Data',alpha=0.7)

plt.title('Prediksi populasi dalam 1000 hari ke depan')

plt.legend();

plt.savefig('Prediction-Now-1000daysFuture.png')
model.save('population-rnn.h5')