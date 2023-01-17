import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
df = pd.read_csv('/kaggle/input/sunspots/Sunspots.csv')

print(df.shape)
df.head()
time = df.iloc[:, 0]
series = df.iloc[:, 2]

print(time.shape)
print(series.shape)
def plot_series(time, series, fmt='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], fmt)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid('on')
plt.figure(figsize=(15, 5))
plot_series(time, series)
plt.figure(figsize=(10, 6))
plot_series(time, series, start=1000, end=1300)
split_time = 3000
time_train = time[:split_time]
X_train = series[:split_time]
time_val = time[split_time:]
X_val = series[split_time:]

plt.figure(figsize=(15, 5))
plot_series(time_train, X_train)
plot_series(time_val, X_val)
plt.legend(['Train', 'Validation'])
def windowed_dataset(series, window_size, batch_size, shuffle=True, shuffle_buffer=None):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)   # step once and slice series into (window_shape + 1) windows. [+1 for output]
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))  # convert them into tensors
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)  # shuffling windows to get rid of 'sequence bias'
    dataset = dataset.map(lambda window: (window[:-1], window[1:]))   # making (x, y) split
    dataset = dataset.batch(batch_size).prefetch(1)   # batching (x, y) into batch_size sets
    return dataset
def model_forecast(model, series, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size)) 
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)
    return forecast
# Hyperparameters
window_size = 64
train_batch_size = 256
val_batch_size = 32
tf.keras.backend.clear_session()
train_set = windowed_dataset(X_train, window_size, train_batch_size, shuffle_buffer=len(X_train))
val_set = windowed_dataset(X_val, window_size, val_batch_size, shuffle=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                          strides=1, padding='causal',
                          activation='relu',
                          input_shape=[None, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer  =tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=['mae'])
history = model.fit(train_set, 
                    epochs=100,
                    validation_data=val_set,
                    callbacks=[lr_scheduler])
# Training, Validation loss & mae
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Training', 'Validation'])
plt.title('Training & Validation Loss')

plt.subplot(122)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.xlabel('epochs')
plt.ylabel('mae')
plt.legend(['Training', 'Validation'])
plt.title('Training & Validation MAE')
plt.show()
# plotting learning-rate vs loss
plt.semilogx(history.history['lr'], history.history['loss'])
plt.axis([1e-8, 1e-3, 0, 80])
plt.axvline(8e-6, color='orange', alpha=0.4)
# Hyperparameters
window_size = 60
train_batch_size = 100
val_batch_size = 32
tf.keras.backend.clear_session()
train_set = windowed_dataset(X_train, window_size, train_batch_size, shuffle_buffer=len(X_train))
val_set = windowed_dataset(X_val, window_size, val_batch_size, shuffle=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding='causal',
                          activation='relu',
                          input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])

optimizer = tf.keras.optimizers.SGD(learning_rate=8e-6, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=['mae'])
history = model.fit(train_set, 
                    epochs=500,
                    validation_data=val_set)
# Training, Validation loss & mae
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Training', 'Validation'])
plt.title('Training & Validation Loss')

plt.subplot(122)
plt.plot(history.history['loss'][200:])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Training', 'Validation'])
plt.title('Zoomed Training Loss')
forecast = model_forecast(model, series[:, np.newaxis], window_size, batch_size=32)
forecast = forecast[split_time-window_size : -1, -1, 0]
plt.figure(figsize=(10, 6))
plot_series(time_val, X_val)
plot_series(time_val, forecast)
tf.keras.metrics.mean_absolute_error(X_val, forecast).numpy()
# Hyperparameters
window_size = 60
train_batch_size = 100
val_batch_size = 32
tf.keras.backend.clear_session()
train_set = windowed_dataset(X_train, window_size, train_batch_size, shuffle_buffer=len(X_train))
val_set = windowed_dataset(X_val, window_size, val_batch_size, shuffle=False)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                          strides=1, padding='causal',
                          activation='relu',
                          input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.5, 
                                                 patience=5, 
                                                 min_lr=1e-7)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', mode='min', patience=15)
optimizer = tf.keras.optimizers.Adam()
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=['mae'])
history = model.fit(train_set, 
                    epochs=500,
                    validation_data=val_set, 
                    callbacks=[reduce_lr, earlystop])
# Training, Validation loss & mae
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Training', 'Validation'])
plt.title('Training & Validation Loss')
forecast = model_forecast(model, series[:, np.newaxis], window_size, batch_size=32)
forecast = forecast[split_time-window_size : -1, -1, 0]
plt.figure(figsize=(10, 6))
plot_series(time_val, X_val)
plot_series(time_val, forecast)
tf.keras.metrics.mean_absolute_error(X_val, forecast).numpy()