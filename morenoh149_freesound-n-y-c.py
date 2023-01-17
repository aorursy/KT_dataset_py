import numpy as np
import pandas as pd
import os
import scipy.io.wavfile
import keras
print(os.listdir("../input"))
audio_paths = []
for line in os.listdir('../input/audio_train'):
    audio_path = '../input/audio_train/' + line.strip()
    audio_paths.append(audio_path)
audio_paths[:10]
x_train = np.random.random((1000, 5))
x_train
y_train = keras.utils.to_categorical([0], num_classes=1)
y_train
fs, data = scipy.io.wavfile.read('../input/audio_train/513f4971.wav')
data
data.shape
x_train = [data]
x_train
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_shape=data.shape[1:]))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)
score = model.evaluate(x_test, y_test, batch_size=128)
