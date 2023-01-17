INPUT_PATH = '../input/'
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.utils import np_utils

from keras.layers.core import Dense, Dropout

from keras.layers import Convolution2D, MaxPooling2D, Flatten

from keras.utils import plot_model, np_utils

from sklearn.model_selection import train_test_split
from subprocess import check_output

print(check_output(["ls", INPUT_PATH]).decode("utf8"))
train = pd.read_csv(INPUT_PATH + 'train.csv')

train.sample(10)
train_y = keras.utils.to_categorical(train.label.values)
train_x = train.drop('label', axis=1).values
def normalize(array):

    # Normalize the data

    return array.astype(np.float32) / 255.0



train_x = normalize(train_x)
train_x = train_x.reshape(train_x.shape[0],28,28,1)
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33, random_state=777)
def get_model():

    model = keras.models.Sequential()

    # Hidden layer of 512 neurons

    model.add(Convolution2D(32,3,3, activation='relu', input_shape=(28,28,1,)))

    model.add(Convolution2D(32,3,3, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))

    return model
model = get_model()

model.summary()



model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y))
score = model.evaluate(test_x, test_y, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
train = pd.read_csv(INPUT_PATH + 'train.csv')

train_y = keras.utils.to_categorical(train.label.values)

train_x = train.drop('label', axis=1).values

train_x = normalize(train_x)

train_x = train_x.reshape(train_x.shape[0],28,28,1)

model = get_model()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=32, epochs=10)
test = pd.read_csv(INPUT_PATH + 'test.csv')

test_x = normalize(test.values)

test_x = test_x.reshape(test_x.shape[0],28,28,1)



predictions = model.predict(test_x)
import time



# select the indix with the maximum probability

predictions = np.argmax(predictions,axis = 1)

predictions = pd.Series(predictions,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)



# Creo una variable con la fecha y hora

datetime = time.strftime("%Y%m%d_%H%M%S")



# Creo el archivo de submission con la fecha actual en el nombre del archivo.

submission.to_csv('submission.' + datetime + '_loss_' + str(score[0]) + '_acc_' + str(score[1]) + '.csv', index=False)