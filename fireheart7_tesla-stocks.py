import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset = pd.DataFrame(pd.read_csv("/kaggle/input/tesla-stock-price/Tesla.csv - Tesla.csv.csv"))
dataset.shape
dataset.head()
dataset.tail()
# check for any correlation

plt.figure(figsize = (10,10))

sns.heatmap(dataset.corr(), annot = True, fmt = ".1g", vmin = -1, vmax = 1, center = 0, linewidth = 3,

           linecolor = "black", square = True, cmap = "summer")
dataset.info()
plt.figure(figsize = (20, 12))

x = np.arange(0, dataset.shape[0], 1)

plt.subplot(2,1,1)

plt.plot(x, dataset.Open.values, color = "red", label = "Open Tesla Price")

plt.plot(x, dataset.Close.values, color = "blue", label = "Close Tesla Price")

plt.title("Tesla Stock Prices 2010-2017")

plt.xlabel("Days")

plt.ylabel("Stock Prices in US Dollar")

plt.legend(loc = "best")

plt.grid(which = "major", axis = "both")



plt.subplot(2,1,2)

plt.plot(x, dataset.Volume.values, color = "green", label = "Stock Volume Available")

plt.title("Stock Volume of Tesla b/w 2010-2017")

plt.xlabel("Days")

plt.ylabel("Volume")

plt.legend(loc = "best")

plt.grid(which = "major", axis = "both")

plt.show()
TIME_STEP = 5

DAYS = 20 # number of days at the end for which we have to predict. These will be in our validation set.
dataset = pd.DataFrame(pd.read_csv("/kaggle/input/tesla-stock-price/Tesla.csv - Tesla.csv.csv"))
def dataset_split(dataset) : 

    train = dataset[0: len(dataset) - DAYS]

    val = dataset[len(dataset) - DAYS - TIME_STEP : len(dataset)]

    return train, val
dataset.drop(["Date","High", "Low", "Close", "Volume", "Adj Close"], axis = 1, inplace = True)

dataset = dataset.values
import sklearn

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range = (0,1))

dataset_scaled = scaler.fit_transform(dataset)
train, val = dataset_split(dataset_scaled)
train.shape, val.shape
train_x, train_y = [], []

for i in range(TIME_STEP, train.shape[0]) : 

    train_x.append(train[i - TIME_STEP : i, 0])

    train_y.append(train[i, 0])

train_x, train_y = np.array(train_x), np.array(train_y)
val_x, val_y = [], []

for i in range(TIME_STEP, val.shape[0]) : 

    val_x.append(val[i - TIME_STEP : i, 0])

    val_y.append(val[i, 0])

val_x, val_y = np.array(val_x), np.array(val_y)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))

print("Reshaped train_x = ", train_x.shape)

print("Shape of train_y = ", train_y.shape)



print("Reshaped val_x = ", val_x.shape)

print("Shape of val_y = ", val_y.shape)
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices("GPU")

print(gpus)

if len(gpus) == 1 : 

    strategy = tf.distribute.OneDeviceStrategy(device = "/gpu:0")

else:

    strategy = tf.distribute.MirroredStrategy()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision" : True})

print("Mixed precision enabled")
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor= "loss", factor = 0.5, patience = 10,

                                                 min_lr = 0.000001, verbose = 1)

monitor_es = tf.keras.callbacks.EarlyStopping(monitor= "loss", patience = 25, restore_best_weights= False, verbose = True)
model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(units = 128, return_sequences = True, input_shape = (train_x.shape[1], 1)))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.LSTM(units = 128, return_sequences = True))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.LSTM(units = 128, return_sequences = True))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.LSTM(units = 128, return_sequences = False))

model.add(tf.keras.layers.Dropout(0.4))



model.add(tf.keras.layers.Dense(units = 20, activation = "relu"))

model.add(tf.keras.layers.Dense(units = 1, activation = "relu"))
model.compile(tf.keras.optimizers.Adam(lr = 0.001), loss = "mean_squared_error")
model.summary()
with tf.device("/device:GPU:0"):

    history = model.fit(train_x, train_y, epochs = 300, batch_size = 16, callbacks = [reduce_lr, monitor_es])
plt.figure(figsize = (12, 4))

plt.plot(history.history["loss"], label = "Training loss")

plt.title("Loss analysis")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train"])

plt.grid("both")
model_json = model.to_json()

with open("tesla_open_1.json", "w") as json_file:

  json_file.write(model_json)



model.save_weights("tesla_open_1.h5")
from keras.models import model_from_json

json_file = open('tesla_open_1.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("tesla_open_1.h5")

print("Loaded model from disk")

loaded_model.compile(loss='mean_squared_error', optimizer='adam')
real_prices = val[TIME_STEP:]

real_prices = scaler.inverse_transform(real_prices)
predicted_prices = loaded_model.predict(val_x)

predicted_prices = scaler.inverse_transform(predicted_prices)
plt.figure(figsize= (16, 5))

plt.subplot(1,1,1)



x = np.arange(0, DAYS, 1)



plt.plot(x, real_prices, color = "red", label = "Real Tesla Prices")

plt.plot(x, predicted_prices, color = "blue", label = "Predicted Tesla Prices")

plt.title("Tesla Open Stock Prices", fontsize = 18)

plt.xlabel("Time In Days", fontsize = 18)

plt.ylabel("Stock Prices in US Dollars", fontsize = 18)

plt.legend()

plt.grid("both")
original_training_prices = scaler.inverse_transform(train)

original_training_prices
x1 = np.arange(0,len(original_training_prices),1)

x2 = np.arange(len(original_training_prices), len(dataset), 1)

print(len(x1), len(x2))
plt.figure(figsize= (16,8))

plt.subplot(1,1,1)



X = len(dataset)

x1 = np.arange(0,len(original_training_prices),1)

x2 = np.arange(len(original_training_prices), len(dataset), 1)



plt.plot(x1, original_training_prices, color = "green")

plt.plot(x2, real_prices, color = "red", label = "Real Tesla Prices")

plt.plot(x2, predicted_prices, color = "blue", label = "Predicted Tesla Prices")

plt.title("Tesla Open Stock Prices", fontsize = 18)

plt.xlabel("Time In Days", fontsize = 18)

plt.ylabel("Stock Prices in US Dollars", fontsize = 18)

plt.legend()

plt.grid("both")