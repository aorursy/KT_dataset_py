import numpy as np

import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt
time = np.arange(365*4 + 1, dtype="float32")
time
def trend(time, slope=0):

    return slope * time



def seasonal_pattern(season_time):

    """Just an arbitrary pattern, you can change it if you wish"""

    return np.where(season_time < 0.4,

                    np.cos(season_time * 2 * np.pi),

                    1 / np.exp(3 * season_time))



def seasonality(time, period, amplitude=1, phase=0):

    """Repeats the same pattern at each period"""

    season_time = ((time + phase) % period) / period

    return amplitude * seasonal_pattern(season_time)



def noise(time, noise_level=1, seed=None):

    rnd = np.random.RandomState(seed)

    return rnd.randn(len(time)) * noise_level

    

baseline = 10

slope = 0.09

amplitude = 20

noise_level = 5
series = baseline + trend(time=time, slope=slope) + seasonality(time=time, period=365, amplitude=amplitude)

series += noise(time=series, noise_level=noise_level)
# take a look on dataset(time series data)

plt.figure(figsize=(10, 10))

plt.title("Series Value at each Time")

plt.xlabel("Time")

plt.ylabel("Value")

plt.plot(time, series)
# split dataset(time and series) into train and validation

time_train = time[:1000]

series_train = series[:1000]



time_validation = time[1000:]

series_validation = series[1000:]
# take a look on time series data for training

plt.figure(figsize=(10, 10))

plt.title("Series Value at each Time")

plt.xlabel("Time")

plt.ylabel("Value")

plt.plot(time_train, series_train)
# take a look on time series data for validation

plt.figure(figsize=(10, 10))

plt.title("Series Value at each Time")

plt.xlabel("Time")

plt.ylabel("Value")

plt.plot(time_validation, series_validation)
# create SliceDataset

dataset = tf.data.Dataset.from_tensor_slices(series_train)

dataset
# iterate

for value in dataset:

    print(value)
# create WindowDataset

dataset = dataset.window(size=21, shift=1, drop_remainder=True)

dataset
# iterate

for window in dataset:

    print(window)
# iterate again

for window in dataset:

    for value in window:

        print(value)

    print()
# create MapDataset

dataset = dataset.map(lambda window: window.batch(21))

dataset
# iterate

for batch in dataset:

    print(batch)
# iterate again

for batch in dataset:

    for value in batch:

        print(value)

    print()
# create FlatMapDataset

dataset = dataset.flat_map(lambda batch: batch)

dataset
# iterate

for batch in dataset:

    print(batch)

    print()
# create ShuffleDataset

dataset = dataset.shuffle(1000)

dataset
# iterate

for batch in dataset:

    print(batch)

    print()
# create MapDataset

dataset = dataset.map(lambda batch: (batch[:-1], batch[-1]))

dataset
# iterate

for batch in dataset:

    print(batch)

    print()
# create BatchDataset

dataset = dataset.batch(32)

dataset
# iterate

for batch_ in dataset:

    print(batch_)

    print()
# create Prefatch Dataset

dataset = dataset.prefetch(1)

dataset
model = keras.Sequential()
model.add(keras.layers.Dense(units=20, activation="relu", input_shape=(20, )))

model.add(keras.layers.Dense(units=10, activation="relu"))

model.add(keras.layers.Dense(units=1, activation="linear"))
model.summary()
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.0001))
model.fit(dataset,

          epochs=200)
predicted_value = []
for i in range(len(series) - 20):

    x = series[i:i+20]

    x = np.reshape(x, (1, -1))

    y = model.predict(x)[0][0]

    predicted_value.append(y)
predicted_value_ = list(range(20)) + predicted_value

predicted_value_ = predicted_value_[1000:]
# compare predicted_value with series_validation

plt.figure(figsize=(12, 12))

plt.title("Compare Predicted Value with Validation Value")

plt.xlabel("Time")

plt.ylabel("Value")

plt.plot(time_validation, series_validation, color="blue", label="Validation")

plt.plot(time_validation, predicted_value_, color="red", label="Predicted")

plt.legend()
keras.metrics.mean_squared_error(y_true=series_validation, y_pred=predicted_value_)