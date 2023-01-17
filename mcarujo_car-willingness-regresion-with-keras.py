!pip install nb_black -q
%load_ext nb_black
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly



data = pd.read_csv(

    "../input/car-purchase-data/Car_Purchasing_Data.csv", encoding="ISO-8859-1"

)

data.head()
sns.pairplot(data)
dc = data

dc.drop(["Customer Name", "Customer e-mail", "Country"], inplace=True, axis=1)

dc.head(2)
x = dc[["Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"]].values

y = dc["Car Purchase Amount"].values.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler

from tabulate import tabulate



scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)

print(

    tabulate(

        [

            ["Type", "Gender", "Age", "Annual Salary", "Credit Card Debt", "Net Worth"],

            ["max"] + list(scaler.data_max_),

            ["min"] + list(scaler.data_min_),

            ["range"] + list(scaler.data_range_),

        ],

        tablefmt="psql",

    )

)
scaler = MinMaxScaler()

y_scaled = scaler.fit_transform(y)

print(

    tabulate(

        [

            ["Type", "Car Purchase Amount"],

            ["max"] + list(scaler.data_max_),

            ["min"] + list(scaler.data_min_),

            ["range"] + list(scaler.data_range_),

        ],

        tablefmt="psql",

    )

)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled)

print("x_train ->", x_train.shape)

print("x_test ->", x_test.shape)

print("y_train ->", y_train.shape)

print("y_test ->", y_test.shape)
import tensorflow.keras

from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(25, input_dim=5, activation="relu"))

model.add(Dense(25, activation="relu"))

model.add(Dense(25, activation="relu"))

model.add(Dense(25, activation="relu"))

model.add(Dense(25, activation="relu"))

model.add(Dense(1, activation="linear"))



model.summary()
model.compile(optimizer="adam", loss="mean_squared_error")



epochs_hist = model.fit(

    x_train, y_train, epochs=30, batch_size=25, verbose=1, validation_split=0.2

)
plt.figure(figsize=(10, 5))

plt.plot(epochs_hist.history["loss"])

plt.plot(epochs_hist.history["val_loss"])

plt.title("Model Loss Progress during training")

plt.ylabel("Loss Value")

plt.xlabel("Epochs")

plt.legend(["Training Loss", "Validation Loss"])
y_pred = model.predict(x_test)

plt.figure(figsize=(15, 8))

plt.plot(y_test)

plt.plot(y_pred)

plt.title("Model Prediction")

plt.ylabel("Car Purchase Amount")

plt.xlabel("Sample")

plt.legend(["Real Value", "Predict Value"])
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



print("r2 score in our model: ", r2_score(y_test, y_pred))

print("mean squared error in our model: ", mean_squared_error(y_test, y_pred))