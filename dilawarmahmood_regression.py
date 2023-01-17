!pip install -q git+https://github.com/tensorflow/docs
import tensorflow as tf

from tensorflow import keras

import tensorflow_docs as tfdocs

import tensorflow_docs.modeling

import tensorflow_docs.plots



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('../input/autompg-dataset/auto-mpg.csv', na_values="?")

data.tail()
data.pop("car name")
data.isna().sum()
data = data.dropna()
data["origin"] = data["origin"].map({1: "usa", 2: "europe", 3: "japan"})
one_hot = pd.get_dummies(data["origin"], prefix="", prefix_sep="")

one_hot
data.pop("origin")

data = pd.concat([data, one_hot], axis=1)

data.tail()
train_data = data.sample(frac=0.8, random_state=0)

test_data = data.drop(train_data.index)
sns.pairplot(train_data[["mpg", "cylinders", "displacement", "weight"]], diag_kind="kde")
train_stats = train_data.describe()

train_stats.pop("mpg")

train_stats = train_stats.transpose()

train_stats
train_labels = train_data.pop("mpg")

test_labels = test_data.pop("mpg")
def norm(x):

    return (x - train_stats['mean']) / train_stats['std']
n_train_data = norm(train_data)

n_test_data = norm(test_data)
model = keras.Sequential([

    keras.layers.Dense(64, activation="relu", input_shape=[len(train_data.keys())]),

    keras.layers.Dense(64, activation="relu"),

    keras.layers.Dense(1)

])



model.summary()
model.compile(optimizer=keras.optimizers.RMSprop(0.001),

              loss="mse",

              metrics=["mse", "mae"])
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
history = model.fit(n_train_data, train_labels,

                    epochs=1000, validation_split=0.2, verbose=0,

                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])
hist = pd.DataFrame(history.history)

hist["epoch"] = history.epoch

hist.tail()
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({"Basic": history}, metric="mae")

plt.ylim([0, 10])

plt.ylabel('MAE [MPG]')
plotter.plot({"Basic": history}, metric="mse")

plt.ylim([0, 20])

plt.ylabel('MSE [MPG^2]')
loss, mse, mae = model.evaluate(n_test_data, test_labels, verbose=2)



print("Testing set Mean Abs Error: {:5.2f} MGP".format(mae))
predictions = model.predict(n_test_data).flatten()



a = plt.axes(aspect="equal")

plt.scatter(test_labels, predictions)

plt.xlabel("True Values [MPG]")

plt.ylabel("Predictions [MPG]")

lims = [0, 50]

plt.xlim(lims)

plt.ylim(lims)

_ = plt.plot(lims, lims)
error = predictions - test_labels

plt.hist(error, bins=25)

plt.xlabel("Prediction Error [MPG]")

_ = plt.ylabel("Count")