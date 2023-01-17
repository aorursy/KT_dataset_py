import numpy as np

from calendar import monthrange

import time

import pandas as pd 

import os 

cols = ["Date","Open_price","Max_price","Min_price","Close_price","Volumen"]

df_cd = pd.read_csv("../input/cdr_d.csv", sep = ",", header=None, skiprows = 1, names = cols)

df_cd.head
df_cd.dtypes

df_cd = df_cd.drop(['Date'],axis = 1)

df_cd = df_cd.drop(['Volumen'],axis = 1)
df_final = df_cd
mean = df_cd.mean(axis=0)

std = df_cd.std(axis=0)

df_final -=mean

df_final/=std

df_final.head
df_final_np = df_final.to_numpy()

df_final_np.shape
def generator(data,lookback, min_index, max_index, shuffle = False, batch_size = 128, step = 1 ):

    if max_index is None:

        max_index = len(data)-1

    i = min_index + lookback

    while 1:

        if shuffle:

            rows = np.random.randint(min_index + lookback, max_index, size = batch_size)

        else:

            if i + batch_size >= max_index:

                i = min_index + lookback

            rows = np.arange(i,min(i + batch_size, max_index))

            i += len(rows)

        samples = np.zeros((len(rows), lookback//step, data.shape[-1]))

        

        targets = np.zeros((len(rows),))

        

        for j, row in enumerate(rows):

            indices = range(rows[j]-lookback, rows[j], step)

            samples[j] = data[indices]

            targets[j] = data[rows[j]][3]

        yield samples, targets 

            

            

            

            

            
lookback  = 10 # our days which we take into consideration to predict next day closing price 

step = 1 #future day 

batch_size = 50 # how many training examples in the batch
train_generator = generator(df_final_np, lookback = lookback, min_index = 0, max_index = 4000, shuffle = True, step = step, batch_size = batch_size)

val_generator = generator(df_final_np, lookback = lookback, min_index = 4001, max_index = 6000, shuffle = False, step = step, batch_size = batch_size)

test_generator = generator(df_final_np, lookback = lookback, min_index = 6001, max_index = None, shuffle = False, step = step, batch_size = batch_size)



val_steps = (6000 - 4001 - lookback)//batch_size

test_steps = (len(df_final_np)- 6001 - lookback)//batch_size
val_steps
test_steps
from keras.models import Sequential

from keras import layers

from keras.optimizers import Adagrad, RMSprop



model = Sequential()

model.add(layers.GRU(32,input_shape = (None, df_final_np.shape[-1])))

model.add(layers.Dense(1))



model.compile(optimizer = Adagrad(), loss = 'mae')



history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 20, validation_data = val_generator, validation_steps = val_steps)

import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(loss))



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Train loss')

plt.plot(epochs, val_loss, 'b', label = 'Val loss')

plt.legend()



print(plt.show())

model_f = Sequential()

model_f.add(layers.LSTM(32,input_shape = (None, df_final_np.shape[-1])))

model_f.add(layers.Dense(1))



model_f.compile(optimizer = Adagrad(), loss = 'mae')



history_f = model_f.fit_generator(train_generator, steps_per_epoch = 100, epochs = 20, validation_data = val_generator, validation_steps = val_steps)
import matplotlib.pyplot as plt



loss = history_f.history['loss']

val_loss = history_f.history['val_loss']



epochs = range(len(loss))



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Train loss')

plt.plot(epochs, val_loss, 'b', label = 'Val loss')

plt.legend()



print(plt.show())
model = Sequential()

model.add(layers.LSTM(32,return_sequences=True,input_shape = (None, df_final_np.shape[-1])))

model.add(layers.LSTM(64, activation='relu'))

model.add(layers.Dense(1))

model.compile(optimizer = Adagrad(), loss = 'mae')



history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 20, validation_data = val_generator, validation_steps = val_steps)
import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(loss))



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Train loss')

plt.plot(epochs, val_loss, 'b', label = 'Val loss')

plt.legend()



print(plt.show())
model = Sequential()

model.add(layers.LSTM(32,dropout=0.2,recurrent_dropout=0.2,input_shape = (None, df_final_np.shape[-1])))

model.add(layers.Dense(1))



model.compile(optimizer = Adagrad(), loss = 'mae')



history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 20, validation_data = val_generator, validation_steps = val_steps)
import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(loss))



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Train loss')

plt.plot(epochs, val_loss, 'b', label = 'Val loss')

plt.legend()



print(plt.show())
model = Sequential()

model.add(layers.LSTM(32,dropout=0.2,recurrent_dropout=0.2,return_sequences=True,input_shape = (None, df_final_np.shape[-1])))

model.add(layers.LSTM(64,dropout=0.2,recurrent_dropout=0.2, activation='relu'))

model.add(layers.Dense(1))

model.compile(optimizer = Adagrad(), loss = 'mae')



history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 20, validation_data = val_generator, validation_steps = val_steps)
import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(loss))



plt.figure()



plt.plot(epochs, loss, 'bo', label = 'Train loss')

plt.plot(epochs, val_loss, 'b', label = 'Val loss')

plt.legend()



print(plt.show())
predict = model_f.predict_generator(test_generator,steps = test_steps)

model_f.evaluate_generator(test_generator,steps = test_steps)