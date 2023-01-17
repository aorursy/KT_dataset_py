import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt
# collect spread ratio of two stocks as numpy array

close1= pd.read_csv("../input/pair-trading-dataset/F.csv").iloc[-2490:-1, :]["Close"].values

close2 = pd.read_csv("../input/pair-trading-dataset/GM.csv").iloc[-2490:-1, :]["Close"].values

spread = close1 / close2
spread, spread.shape
time = np.array(list(range(len(spread))))

series = spread
# split dataset into train and validation part

time_train = time[:2000]

series_train = series[:2000]

series_train = np.reshape(series_train, (-1, 1))



time_validation = time[2000:]

series_validation = series[2000:]
series_train.shape, series_validation.shape
# convert numpy array to Tensorflow Dataset object

dataset = tf.data.Dataset.from_tensor_slices(series_train)

dataset
# split dataset into many windows

dataset = dataset.window(size=21, shift=1, drop_remainder=True)

dataset
# convert each window to a batch

dataset = dataset.map(lambda window: window.batch(21))

dataset
# flatten each batch

dataset = dataset.flat_map(lambda batch: batch)

dataset
# shuffle the order of batch

dataset = dataset.shuffle(buffer_size=2000)

dataset
# split each batch into input and output

dataset = dataset.map(lambda batch: (batch[:-1], batch[-1]))

dataset
# integrate 32 batchs/examples into a "batch" for training

dataset = dataset.batch(32)

dataset
# prefatch Tensorflow Dataset object to enhance efficiency of training process

train_dataset = dataset.prefetch(1)

train_dataset
for b in train_dataset:

    print(b)

    print()
model = keras.models.Sequential()

model.add(keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu", input_shape=(20, 1)))

# model.add(keras.layers.Dense(units=128, input_shape=(20, 1), activation="relu"))

model.add(keras.layers.LSTM(64, return_sequences=True))

model.add(keras.layers.LSTM(64))

model.add(keras.layers.Dense(30, activation="relu"))

model.add(keras.layers.Dense(30, activation="relu"))

model.add(keras.layers.Dense(1, activation="linear"))
model.summary()
model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=["mae"])
model.fit(x=train_dataset, epochs=100, verbose=1)
ans = []



for idx in range(2000, len(series)):

    arr = series[idx-20:idx]

    arr = np.reshape(arr, (1, 20, 1))

    out = model.predict(arr)[0]

    ans.append(out)
plt.figure(figsize=(15, 10))

plt.plot(time_validation, ans, color="red")

plt.plot(time_validation, series_validation, color="blue")
from sklearn.metrics import mean_absolute_error
mean_absolute_error(ans, series_validation)