!pip install nb_black -q
%load_ext nb_black
import warnings



warnings.simplefilter("ignore", category=FutureWarning)
import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from keras.datasets import cifar10



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
from keras import utils



x_train = x_train / 225

x_test = x_test / 225



y_train_cat = utils.to_categorical(y_train, 10)

y_test_cat = utils.to_categorical(y_test, 10)



input_shape = x_train.shape[1:]

output_shape = y_train_cat.shape[1]
print(x_train.shape)

print(y_train_cat.shape)
fig, axes = plt.subplots(10, 10, figsize=(28, 28))

axes = axes.ravel()



for i in np.arange(0, 100):

    axes[i].imshow(x_train[i])

    axes[i].set_title(y_train[i])

    axes[i].axis("off")
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard





model = Sequential()

model.add(

    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=input_shape)

)

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.1))



model.add(Flatten())

model.add(Dense(units=1024, activation="relu"))

model.add(Dense(units=1024, activation="relu"))

model.add(Dense(units=output_shape, activation="softmax"))



model.summary()
model.compile(

    loss="categorical_crossentropy",

    optimizer=Adam(lr=0.001),

    metrics=["accuracy", "mse"],

)





history = model.fit(x_train, y_train_cat, epochs=30, shuffle=True)
import plotly.express as px



df_mse = pd.DataFrame(history.history["mse"])

df_mse["1"] = "MSE"

df_mse["Epoch"] = df_mse.index



df_loss = pd.DataFrame(history.history["loss"])

df_loss["1"] = "Loss"

df_loss["Epoch"] = df_loss.index



df_accuracy = pd.DataFrame(history.history["accuracy"])

df_accuracy["1"] = "Accuracy"

df_accuracy["Epoch"] = df_accuracy.index



df = pd.concat([df_accuracy, df_loss, df_mse])

df.columns = ["Value", "Metric", "Epoch"]



fig = px.line(df, x="Epoch", y="Value", color="Metric")

fig.show()
evaluation = model.evaluate(x_test, y_test_cat)

print("Test Accuracy {}".format(evaluation[1]))
from sklearn.metrics import confusion_matrix



m_c = confusion_matrix(y_test, model.predict(x_test).argmax(1))

plt.figure(figsize=(15, 14))

sns.heatmap(m_c, annot=True, cmap="Reds", fmt="d").set(xlabel="Predict", ylabel="Real")