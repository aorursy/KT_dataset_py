import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



airpassengers = pd.read_csv("/kaggle/input/airpassenger.csv")

airpassengers.columns = ["Time","Passengers"]

airpassengers.head()
plt.figure(figsize=(16,8))

plt.plot(range(airpassengers.shape[0]), airpassengers["Passengers"].values, color='tab:blue')

plt.gca().set(title="Airpassengers", xlabel="Time", ylabel="Passengers")

plt.show()
airpassengers_array = airpassengers["Passengers"].values
def split_sequence(sequence, n_steps=12):

    X, y = list(), list()

    for i in range(len(sequence)):

        end_ix = i + n_steps

        if end_ix > len(sequence)-1:

            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)



X, y = split_sequence(airpassengers_array)
for i in range(len(X)):

    print(X[i], y[i])
X, y = split_sequence(airpassengers_array, n_steps=3)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
model = Sequential()

model.add(Dense(12, activation='relu', input_dim=3))

model.add(Dense(8, activation='relu'))

model.add(Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs = 250)
yhat = model.predict(X)
from sklearn.metrics import r2_score

r2_score(y, yhat)
plt.figure(figsize=(16,8))

plt.plot(range(len(y)), y, color='tab:blue')

plt.plot(range(len(yhat)), yhat, color='tab:red')

plt.gca().set(title="Airpassengers", xlabel="Time", ylabel="Passengers")

plt.show()