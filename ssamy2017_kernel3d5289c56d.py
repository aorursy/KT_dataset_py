import numpy as np;

import pandas as pd;

import seaborn as sb;

import sklearn as sk;

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/traningdata.csv")
dataset = dataset.drop(columns = 'Unnamed: 0')

dataset.describe()
X= dataset[['alpha','beta','nu','rho','F','K','T']]

Y= dataset['vola']

X_train, X_valtest, y_train, y_valtest = train_test_split(X, Y, test_size=0.3, random_state=0)

X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5)
FFNN = Sequential()
FFNN.add(Dense(14, input_dim=7, activation='relu'))

FFNN.add(Dense(7,  activation='relu'))

FFNN.add(Dense(1))
FFNN.compile(loss='mean_squared_error', optimizer='adam')
hist = FFNN.fit(X_train, y_train, epochs=50, batch_size=500,validation_data=(X_val, y_val))
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()
y_pred = FFNN.predict(X_test)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms_NN = sqrt(mean_squared_error(y_test, y_pred))
print (rms_NN)


FFNN.save('FFNN.h5')