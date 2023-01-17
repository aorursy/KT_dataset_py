import keras

import pandas

import numpy as np

from keras.models import *

from keras.layers import *

from keras.optimizers import *



data_full = pandas.read_csv('../input/training.csv',  dtype={'label':'category'})

x = data_full.iloc[:,1:]

y = data_full.iloc[:,0]



batch_size = 30

num_classes = 10

epochs = 20

input_shape = (40,)



x_train = np.array(x[:350])

x_test = np.array(x[350:])

y_train = keras.utils.to_categorical(y[:350], num_classes)

y_test = keras.utils.to_categorical(y[350:], num_classes)



model = Sequential()

model.add(Dense(40, input_shape=input_shape, activation='relu'))

model.add(Dense(40, input_shape=input_shape, activation='relu'))

model.add(Dense(40, input_shape=input_shape, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_split=1/4)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
