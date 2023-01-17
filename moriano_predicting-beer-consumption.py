# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw = pd.read_csv("../input/Consumo_cerveja.csv", decimal=",")
raw.head()
raw.columns=["date", "avgTemp", "minTemp", "maxTemp", "rain", "weekend", "litres"]
raw.head()
raw.dtypes
raw["litres"] = raw["litres"].astype(float)
raw = raw.dropna()
raw.dtypes
raw.describe()
raw.plot(kind='scatter', x="maxTemp", y="litres", title = "Consumption and average temperature")
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.subplot(2, 2, 1)
plt.scatter( 'avgTemp', 'litres', data=raw, marker='.', color='red', linewidth=1)
plt.xlabel("Average temperature")
plt.ylabel("litres")

plt.subplot(2, 2, 2)
plt.scatter( 'maxTemp', 'litres', data=raw, marker='.', color='red', linewidth=1)
plt.xlabel("Max Temperature")
plt.ylabel("litres")

plt.subplot(2, 2, 3)
plt.scatter( 'rain', 'litres', data=raw, marker='.', color='red', linewidth=1)
plt.xlabel("rain")
plt.ylabel("litres")

plt.subplot(2, 2, 4)
plt.scatter( 'weekend', 'litres', data=raw, marker='.', color='red', linewidth=1)
plt.xlabel("weekend")
plt.ylabel("litres")

del raw["date"]
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
scaler.fit(raw)
y = raw["litres"]
del raw["litres"]
X = raw
print("X shape", X.shape)
print("y shape", y.shape)
X_normalized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression
y_train.shape
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_hat = linear_model.predict(X_test)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def show_metrics(model):
    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)
    print("Train set")
    print("\tMSE", mean_squared_error(y_train, y_train_hat))
    print("\tMAE", mean_absolute_error(y_train, y_train_hat))
    print("Test set")
    print("\tMSE", mean_squared_error(y_test, y_test_hat))
    print("\tMAE", mean_absolute_error(y_test, y_test_hat))

show_metrics(linear_model)
from  sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor()

forest_model.fit(X_train, y_train)
show_metrics(forest_model)
from sklearn.neural_network import MLPRegressor

nn_model = MLPRegressor(hidden_layer_sizes=(64, ), max_iter=2000, verbose=True, learning_rate_init=0.001)
nn_model.fit(X_train, y_train)
show_metrics(nn_model)
raw = pd.read_csv("../input/Consumo_cerveja.csv", decimal=",", parse_dates=["Data"])
raw.columns=["date", "avgTemp", "minTemp", "maxTemp", "rain", "weekend", "litres"]
raw = raw.dropna()

names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for i, x in enumerate(names):
    raw[x] = (raw["date"].dt.dayofweek == i).astype(int)

del raw["date"]
del raw["weekend"]
raw["litres"] = raw["litres"].astype(float)
raw.head()
scaler = MinMaxScaler()
scaler.fit(raw)
y = raw["litres"]
del raw["litres"]
X = raw

print(X.shape)
X_normalized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_hat = linear_model.predict(X_test)
show_metrics(linear_model)
forest_model = RandomForestRegressor()

forest_model.fit(X_train, y_train)
show_metrics(forest_model)
nn_model = MLPRegressor(hidden_layer_sizes=(64, ), max_iter=2000, verbose=True, learning_rate_init=0.001)

nn_model = MLPRegressor(hidden_layer_sizes=(32, ), max_iter=2000, verbose=True, learning_rate_init=0.001)
nn_model.fit(X_train, y_train)
show_metrics(nn_model)
