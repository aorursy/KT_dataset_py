__author__ = 'Marcelo dos Santos'

__version__ = '1.0'

import pandas as pd

df_train = pd.read_csv('../input/X_train.csv')



print('TRAIN DATA')

x_train = df_train.values[:, 3:]

y_train = df_train.values[:, 1]

print(y_train.shape)

print(x_train.shape)



print('TEST DATA')

df_test = pd.read_csv('../input/X_test.csv')

x_test = df_test.values[:, 3:]

y_test = df_test.values[:, 1]

print(y_test.shape)

print(x_test.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation



model = Sequential()

model.add(Dense(64, input_shape=x_train.shape[1:], activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])



model.fit(x_train, y_train, epochs=30, batch_size=64,validation_data=(x_test, y_test))
