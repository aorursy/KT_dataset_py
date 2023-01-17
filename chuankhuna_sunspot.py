# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/sunspots/Sunspots.csv")
df.head()
df.drop(columns=[df.columns[0]], axis=1, inplace=True)
df.rename(columns={
    df.columns[1]: 'monthly_sunspot'
},inplace=True)
df.head()
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='paper', style='whitegrid', rc={'figure.figsize':(8, 5), 'figure.dpi': 120})
google = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58"]
sns.set_palette(google)
df.info()
sample_size = len(df)
time_x = np.arange(sample_size)

# sunspot period
period = 11*12
xtick_period = np.arange(0, sample_size+period, period)
sns.lineplot(time_x, df['monthly_sunspot'])
plt.xticks(xtick_period, rotation=45)
plt.show()
sns.lineplot(time_x, df['monthly_sunspot'])
plt.xticks(xtick_period, rotation=45)
plt.xlim([0, period*5])
plt.show()
num_period = sample_size // period
periods_ind = np.arange(0, num_period)
rand_periods_ind = np.random.choice(periods_ind, 4, replace=False)

time_x_period = np.arange(0, period)
for i in rand_periods_ind:
    sns.lineplot(time_x_period, df[i*period:(i+1)*period]['monthly_sunspot'])
plt.show()
sns.distplot(df['monthly_sunspot'])
plt.show()
fig = plt.figure(figsize=(10, 1.5), dpi=100)
sns.boxplot(df['monthly_sunspot'])
plt.show()
def windowed_dataset_X_Y(series, window_size, shuffle_buffer):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda window: (window[:-1], window[-1:]))

    
    X = []
    Y = []
    
    for window in ds:
        x, y = window
        X.append(x.numpy())
        Y.append(y.numpy())
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X, Y
seed_ = 20200218
np.random.seed(seed_)
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, RNN, Activation
from keras.optimizers import *

import tensorflow as tf
tf.random.set_seed(seed_)
df.info()
split_time = 2800

train = df['monthly_sunspot'][:split_time]
val = df['monthly_sunspot'][split_time:]

train.shape, val.shape
window_size = 30
shuffle_buffer = 1000


X_train, y_train = windowed_dataset_X_Y(train, window_size, shuffle_buffer)
X_val, y_val = windowed_dataset_X_Y(val, window_size, shuffle_buffer)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)
X_train.shape, y_train.shape
X_val.shape, y_val.shape
model = Sequential()
model.add(LSTM(1, input_shape=(window_size, 1),  return_sequences=False))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.add(keras.layers.Lambda(lambda x: x*10))
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, n_epoch):
        self.n_epoch = n_epoch
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == 0 or (epoch+1)%self.n_epoch == 0:
            print(f"Epoch: {epoch+1}")
            for key, value in logs.items():
                print(f"{key}: {value:.4f}", end=" \t ")
            print()
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mean_absolute_error')
batch_size = 32

hist = model.fit(X_train, y_train, epochs=500, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0,
                callbacks=[CustomCallback(25)])
loss = hist.history['loss']
loss_val = hist.history['val_loss']
t = np.arange(1, len(loss)+1)
sns.lineplot(t, loss, label='loss')
t_val = np.arange(1, len(loss_val)+1)
sns.lineplot(t_val, loss_val, label='val loss')
plt.show()
X_train[0]
model.predict(np.array([X_train[0]]))
y_train[0]
forecast = []

for time in range(sample_size - window_size):
    x = df['monthly_sunspot'][time:time+window_size]
    x = np.array([x])
    x = x.reshape(x.shape[0], x.shape[1], -1)
    forecast.append(model.predict(x))
forecast = np.array(forecast).squeeze()

forecast.shape
sns.lineplot(time_x, df['monthly_sunspot'], label='actual')
sns.lineplot(time_x[window_size:], forecast, label='forecast')
plt.show()
sns.lineplot(time_x, df['monthly_sunspot'], label='actual')
sns.lineplot(time_x[window_size:], forecast, label='forecast')
plt.title("validation set")
plt.xlim([split_time, sample_size])
plt.show()
