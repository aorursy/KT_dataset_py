# !pip uninstall tensorflow protobuf --yes

# !pip install tf-nightly
import numpy as np

import pandas as pd

from tensorflow import keras

from tensorflow.keras import layers

from datetime import datetime

from matplotlib import pyplot as plt

from matplotlib import dates as md

master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"



df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"

df_small_noise_url = master_url_root + df_small_noise_url_suffix

df_small_noise = pd.read_csv(df_small_noise_url)



df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"

df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix

df_daily_jumpsup = pd.read_csv(df_daily_jumpsup_url)

print(df_small_noise.head())



print(df_daily_jumpsup.head())



def plot_dates_values(data):

    dates = data["timestamp"].to_list()

    values = data["value"].to_list()

    dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]

    plt.subplots_adjust(bottom=0.2)

    plt.xticks(rotation=25)

    ax = plt.gca()

    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")

    ax.xaxis.set_major_formatter(xfmt)

    plt.plot(dates, values)

    plt.show()



plot_dates_values(df_small_noise)

plot_dates_values(df_daily_jumpsup)



def get_value_from_df(df):

    return df.value.to_list()





def normalize(values):

    mean = np.mean(values)

    values -= mean

    std = np.std(values)

    values /= std

    return values, mean, std





# Get the `value` column from the training dataframe.

training_value = get_value_from_df(df_small_noise)



# Normalize `value` and save the mean and std we get,

# for normalizing test data.

training_value, training_mean, training_std = normalize(training_value)

len(training_value)

TIME_STEPS = 288





def create_sequences(values, time_steps=TIME_STEPS):

    output = []

    for i in range(len(values) - time_steps):

        output.append(values[i : (i + time_steps)])

    # Convert 2D sequences into 3D as we will be feeding this into

    # a convolutional layer.

    return np.expand_dims(output, axis=2)





x_train = create_sequences(training_value)

print("Training input shape: ", x_train.shape)

n_steps = x_train.shape[1]

n_features = x_train.shape[2]



keras.backend.clear_session()

model = keras.Sequential(

    [

        layers.Input(shape=(n_steps, n_features)),

        layers.Conv1D(filters=32, kernel_size=15, padding='same', data_format='channels_last',

            dilation_rate=1, activation="linear"),

        layers.LSTM(

            units=25, activation="tanh", name="lstm_1", return_sequences=False

        ),

        layers.RepeatVector(n_steps),

        layers.LSTM(

            units=25, activation="tanh", name="lstm_2", return_sequences=True

        ),

        layers.Conv1D(filters=32, kernel_size=15, padding='same', data_format='channels_last',

            dilation_rate=1, activation="linear"),

        layers.TimeDistributed(layers.Dense(1, activation='linear'))

    ]

)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

model.summary()

history = model.fit(

    x_train,

    x_train,

    epochs=200,

    batch_size=128,

    validation_split=0.1,

    callbacks=[

        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min", restore_best_weights=True)

    ],

)

plt.plot(history.history["loss"], label="Training Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.legend()

# Get train MAE loss.

x_train_pred = model.predict(x_train)

train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)



plt.hist(train_mae_loss, bins=50)

plt.xlabel("Train MAE loss")

plt.ylabel("No of samples")

plt.show()



# Get reconstruction loss threshold.

threshold = np.max(train_mae_loss)

print("Reconstruction error threshold: ", threshold)

# Checking how the first sequence is learnt

plt.plot(x_train[0])

plt.show()

plt.plot(x_train_pred[0])

plt.show()



def normalize_test(values, mean, std):

    values -= mean

    values /= std

    return values





test_value = get_value_from_df(df_daily_jumpsup)

test_value = normalize_test(test_value, training_mean, training_std)

plt.plot(test_value.tolist())

plt.show()



# Create sequences from test values.

x_test = create_sequences(test_value)

print("Test input shape: ", x_test.shape)



# Get test MAE loss.

x_test_pred = model.predict(x_test)

test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

test_mae_loss = test_mae_loss.reshape((-1))



plt.hist(test_mae_loss, bins=50)

plt.xlabel("test MAE loss")

plt.ylabel("No of samples")

plt.show()



# Detect all the samples which are anomalies.

anomalies = (test_mae_loss > threshold).tolist()

print("Number of anomaly samples: ", np.sum(anomalies))

print("Indices of anomaly samples: ", np.where(anomalies))

# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies

anomalous_data_indices = []

for data_idx in range(TIME_STEPS - 1, len(test_value) - TIME_STEPS + 1):

    time_series = range(data_idx - TIME_STEPS + 1, data_idx)

    if all([anomalies[j] for j in time_series]):

        anomalous_data_indices.append(data_idx)

df_subset = df_daily_jumpsup.iloc[anomalous_data_indices, :]

plt.subplots_adjust(bottom=0.2)

plt.xticks(rotation=25)

ax = plt.gca()

xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")

ax.xaxis.set_major_formatter(xfmt)



dates = df_daily_jumpsup["timestamp"].to_list()

dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]

values = df_daily_jumpsup["value"].to_list()

plt.plot(dates, values, label="test data")



dates = df_subset["timestamp"].to_list()

dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]

values = df_subset["value"].to_list()

plt.plot(dates, values, label="anomalies", color="r")



plt.legend()

plt.show()
