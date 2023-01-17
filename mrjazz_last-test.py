import pandas as pd

from keras.layers.core import Dense, Dropout, Activation

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

from sklearn import preprocessing

import matplotlib.pyplot as plt
seq_len = 3

data = pd.read_csv("../input/test.csv", index_col=0)

dataX = data['<OPEN>'].as_matrix()

dataY = data['<ACTION>'].as_matrix()
def normalize(data, low_limit=0, high_limit=1):

    data_min = min(data)

    data_max = max(data)

    k = (high_limit - low_limit) / (data_max - data_min)

    return [(i - data_min) * k + low_limit for i in data]



dataX = normalize(data['<OPEN>'])
X = []

Y = []

for i in range(len(dataX)-seq_len-1):    

    X.append(dataX[i: i + seq_len])

    if dataX[seq_len+i-1] > 1.058:

        Y.append(1) # dataY[seq_len+i-1]

    else:

        Y.append(0)

        

X = np.array(X)

Y = np.array(Y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))
def train():

    model = Sequential()

    model.add(LSTM(3, input_shape=(seq_len, 1)))    

    model.add(Dense(1, activation='softmax'))

    # model.compile(optimizer='adam', loss='mse') 

    # model.compile(loss="mse", optimizer="rmsprop")

    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])    

    history = model.fit(X, Y, epochs=100, verbose=0)

    plt.plot(history.history['loss'])

    plt.show()    

    return model

model = train()
loss = model.evaluate(X, Y, verbose=0)  

for i in range(len(model.metrics_names)):

    print(str(model.metrics_names[i]) + ": " + str(loss[i]))
plt.plot(dataX)

plt.show()
y = model.predict(X)

plt.plot(y)

plt.show()
plt.plot(dataY)

plt.show()