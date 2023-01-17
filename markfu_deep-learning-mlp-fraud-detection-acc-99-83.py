import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense, Dropout

import matplotlib as mpl

import matplotlib.pyplot as plt

import keras as kr

%matplotlib inline 

# specify jupyter display it 
df = pd.read_csv('../input/creditcard.csv')

df.shape
df.head()
from sklearn.model_selection import train_test_split



train, test = train_test_split(df, test_size = 0.3)
x_train = train.iloc[:,1:30].values[:]

y_train = train['Class'].values[:]



x_test = test.iloc[:,1:30].values[:]

y_test = test['Class'].values[:]



print (x_train.shape, y_train.shape)

print (x_test.shape, y_test.shape)

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(29,)))

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='tanh'))

model.add(Dense(2, activation='softmax'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])
history = model.fit(x_train, y_train,

                    batch_size=500,

                    epochs=10,

                    verbose=1,

                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])