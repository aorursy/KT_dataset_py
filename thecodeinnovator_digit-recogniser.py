import pandas as pn
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.utils import np_utils
warnings.resetwarnings()
train_data = pn.read_csv("../input/train.csv")
test_data = pn.read_csv("../input/test.csv")
x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
print(x_train.shape)
print(y_train.shape)
x_train = x_train.astype('float') / 255
y_train = np_utils.to_categorical(y_train)
print(y_train.shape)
recogniser = Sequential()
recogniser.add(Dense(units = 784, input_dim = 784, activation = 'relu'))
recogniser.add(Dense(units = 10, activation = 'softmax'))
recogniser.summary()
recogniser.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
recogniser.fit(x_train, y_train, epochs = 10, batch_size = 100, verbose = 2, validation_split = 0.2)
x_test = test_data.iloc[:, :].values
x_test = x_test.astype('float') / 255
print(x_test.shape)
y_predict = recogniser.predict(x_test)
y_predict = y_predict.argmax(axis = 1)
print(y_predict.shape)
images = list(range(1, 28001))
submission = pn.DataFrame({'ImageID': images, 'Label': y_predict})
submission.to_csv('submission-digit.csv', index = False)