import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import csv
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Lambda,Conv1D,Dropout
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.metrics import mean_absolute_error,mean_squared_error
from tensorflow.keras.losses import Huber
from tensorflow.keras.utils import plot_model
time=[]
sunspots=[]
with open("../input/sunspots/Sunspots.csv") as f:
    reader = csv.reader(f,delimiter=',')
    next(reader)
    for row in reader:
        time.append(row[0])
        sunspots.append(row[2])

series = np.array(sunspots).astype(float)
time = np.array(time).astype(int)
#Plot Time vs Series
def plot_series(time,series):
    plt.title("Variation of Sunspots with Time")
    sns.lineplot(time,series)
    plt.xlabel("Time")
    plt.ylabel("Value")
plt.figure(figsize=(12,6))
plot_series(time,series)
#Autocorrelation Plot
fig,ax = plt.subplots(1,2,figsize=(15,6))
auto = plot_acf(series,ax=ax[0])
partial = plot_pacf(series,ax=ax[1])
plt.show()
split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
split_time
#Parameters
window_size = 60
batch_size = 100
shuffle_buffer = 1000
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    d = tf.data.Dataset.from_tensor_slices(series)
    d = d.window(window_size + 1, shift=1, drop_remainder=True)
    d = d.flat_map(lambda w: w.batch(window_size + 1))
    d = d.shuffle(shuffle_buffer)
    d = d.map(lambda w: (w[:-1], w[1:]))
    d = d.batch(batch_size).prefetch(1)
    return d
def model_forecast(model,series,batch_size,window_size):
    d = tf.data.Dataset.from_tensor_slices(series)
    d = d.window(window_size, shift=1, drop_remainder=True)
    d = d.flat_map(lambda w: w.batch(window_size))
    d = d.batch(batch_size).prefetch(1)
    forecast = model.predict(d)
    return forecast
tf.keras.backend.clear_session()

train = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer)
val = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer)

model = Sequential()
model.add(Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[None]))
model.add(Conv1D(filters=60,kernel_size=5,strides=1,padding='causal',activation='relu'))
model.add(LSTM(120,return_sequences=True))
model.add(LSTM(120,return_sequences=True))
model.add(Dense(60,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1))
model.add(Lambda(lambda x:x*400))

lr_schedule = LearningRateScheduler(lambda epoch : 1e-8 * 10**(epoch / 20))
model.compile(loss=Huber(),optimizer=SGD(lr=1e-8,momentum=0.9),metrics=['mae'])
history = model.fit(train, epochs=100,validation_data=val,callbacks=[lr_schedule])
#Plot for selecting learning rate
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 100])
plt.xlabel("Epochs")
plt.ylabel("Loss")
#Final Model with lr=8e-6

tf.keras.backend.clear_session()

train = windowed_dataset(x_train,window_size,batch_size,shuffle_buffer)
val = windowed_dataset(x_valid,window_size,batch_size,shuffle_buffer)

model = Sequential()
model.add(Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[None]))
model.add(Conv1D(filters=60,kernel_size=5,strides=1,padding='causal',activation='relu'))
model.add(LSTM(120,return_sequences=True))
model.add(LSTM(120,return_sequences=True))
model.add(Dense(60,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1))
model.add(Lambda(lambda x:x*400))

model.compile(loss=Huber(),optimizer=SGD(lr=8e-6,momentum=0.9),metrics=['mae'])
history = model.fit(train, epochs=200,validation_data=val)
#Plotting graphs for mae and loss
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_"+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string,"val_"+string])
    plt.show()

plt.figure(figsize=(12,6))
plot_graphs(history,'mae')
plt.figure(figsize=(12,6))
plot_graphs(history,'loss')
plt.show()
#Forecast
forecast = model_forecast(model,series[..., np.newaxis],batch_size,window_size)
forecast = forecast[split_time - window_size:-1,-1,0]
#Predicted Plot
plt.figure(figsize=(12, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, forecast)
plt.legend(["Actual","Forecast"])
print("Mean Absolute Error: ",mean_absolute_error(x_valid,forecast).numpy())
print("Mean Squared Error:",mean_squared_error(x_valid,forecast).numpy())