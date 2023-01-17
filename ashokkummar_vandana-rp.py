from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras import optimizers

import numpy as np

from sklearn.utils import shuffle

from sklearn import preprocessing

from scipy.io import loadmat
input_data = loadmat('../input/rp-dataset/input_try.mat')

input_data = input_data.get('input')

target_data = loadmat('../input/rp-dataset/target_try.mat')

target_data = target_data.get('A')
input_data = preprocessing.normalize(input_data)

input_data, target_data = shuffle(input_data, target_data)
n = 100

d = 0.3

model2 = Sequential()

model2.add(Dense(n, activation='relu', input_dim=40000))



model2.add(Dense(n, activation='relu'))

model2.add(Dropout(d))



model2.add(Dense(n, activation='relu'))

model2.add(Dropout(d))



model2.add(Dense(n, activation='relu'))

model2.add(Dropout(d))



model2.add(Dense(n, activation='relu'))

model2.add(Dropout(d))



model2.add(Dense(25, activation='softmax'))
# sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)

# model2.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(input_data, target_data, epochs=1000, batch_size=32, validation_split = 0.2)