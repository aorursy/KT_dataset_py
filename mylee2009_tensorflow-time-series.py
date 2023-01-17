import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

tf.random.set_seed(51)

np.random.seed(51)
data = pd.read_csv('/kaggle/input/scooter-rental-data/scooter_rental_data.csv')

data.head()
data['date'] = pd.to_datetime(data['date'], format = '%Y-%m-%d')
data['TDate'] =  data['date'] + pd.to_timedelta(data.hr, unit='h')

data=data.sort_values('TDate').reset_index()

data.head()
data[14986:15000]
series_r = data['registered-users'].to_numpy()

series_g = data['guest-users'].to_numpy()



time =data.index.values.tolist()
split_time = 14986 #changed from 3000 to 2500 to get more validation data

time_train = time[:split_time]

x_train_r = series_r[:split_time]

x_train_g = series_g[:split_time]

time_valid = time[split_time:]

x_valid_r = series_r[split_time:]

x_valid_g = series_g[split_time:]



window_size = 64 

batch_size = 256

shuffle_buffer_size = 1000
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    series = tf.expand_dims(series, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size + 1))

    ds = ds.shuffle(shuffle_buffer)

    ds = ds.map(lambda w: (w[:-1], w[1:]))

    return ds.batch(batch_size).prefetch(1)



tf.keras.backend.clear_session()

window_size = 64

batch_size = 256



train_set_r = windowed_dataset(x_train_r, window_size, batch_size, shuffle_buffer_size)

train_set_g = windowed_dataset(x_train_g, window_size, batch_size, shuffle_buffer_size)



print(train_set_r)

print(x_train_r.shape)
valid_set_r = windowed_dataset(x_valid_r, window_size, batch_size, shuffle_buffer_size)

valid_set_g = windowed_dataset(x_valid_g, window_size, batch_size, shuffle_buffer_size)
model_r = tf.keras.models.Sequential([

  tf.keras.layers.Conv1D(filters=32, kernel_size=5,

                      strides=1, padding="causal",

                      activation="relu",

                      input_shape=[None, 1]),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.Dense(30, activation="relu"),

  tf.keras.layers.Dense(10, activation="relu"),

  tf.keras.layers.Dense(1),

  tf.keras.layers.Lambda(lambda x: x * 400)

])



model_r.summary()
model_g = tf.keras.models.Sequential([

  tf.keras.layers.Conv1D(filters=32, kernel_size=5,

                      strides=1, padding="causal",

                      activation="relu",

                      input_shape=[None, 1]),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.LSTM(64, return_sequences=True),

  tf.keras.layers.Dense(30, activation="relu"),

  tf.keras.layers.Dense(10, activation="relu"),

  tf.keras.layers.Dense(1),

  tf.keras.layers.Lambda(lambda x: x * 400)

])



model_g.summary()
# from keras.models import model_from_json

# serialize model to JSON

model_json = model_r.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
# training for registered users

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint



checkpointer = ModelCheckpoint('model_r.h5', monitor='val_loss', verbose=1, save_best_only=True)



early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')



lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model_r.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])



history_r = model_r.fit(train_set_r, validation_data = valid_set_r, epochs=100, callbacks=[lr_schedule,checkpointer,early_stopper])
# training for guest users

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint



checkpointer = ModelCheckpoint('model_g.h5', monitor='val_loss', verbose=1, save_best_only=True)



early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')



lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model_g.compile(loss=tf.keras.losses.Huber(),

              optimizer=optimizer,

              metrics=["mae"])



history_g = model_g.fit(train_set_g, validation_data = valid_set_g, epochs=100, callbacks=[lr_schedule,checkpointer,early_stopper])
json_file = open('/kaggle/working/model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()



#loaded_model = model_from_json(loaded_model_json)

loaded_model_r = tf.keras.models.model_from_json(loaded_model_json)

loaded_model_g = tf.keras.models.model_from_json(loaded_model_json)



# load weights into new model

loaded_model_r.load_weights("/kaggle/working/model_r.h5")

loaded_model_g.load_weights("/kaggle/working/model_g.h5")



print("Loaded model from disk")
# import matplotlib.pyplot as plt

# plt.semilogx(history.history["lr"], history.history["loss"])

# plt.axis([1e-8, 1e-4, 0, 60])
def model_forecast(model, series, window_size):

    # preparing ds

    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(window_size, shift=1, drop_remainder=True)

    ds = ds.flat_map(lambda w: w.batch(window_size))

    ds = ds.batch(32).prefetch(1)

   # forecast

    forecast = model.predict(ds)

    return forecast



rnn_forecast_r = model_forecast(loaded_model_r, series_r[..., np.newaxis], window_size)

rnn_forecast_r = rnn_forecast_r[split_time - window_size:-1, -1, 0]



rnn_forecast_g = model_forecast(loaded_model_g, series_g[..., np.newaxis], window_size)

rnn_forecast_g= rnn_forecast_g[split_time - window_size:-1, -1, 0]
tf.keras.metrics.mean_absolute_error(x_valid_r, rnn_forecast_r).numpy()
tf.keras.metrics.mean_absolute_error(x_valid_g, rnn_forecast_g).numpy()