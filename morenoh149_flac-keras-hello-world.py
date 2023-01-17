import numpy as np
import soundfile as sf # install this in the kernel under Settings
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
path = '../input/songs/songs/xc101862.flac'
data, samplerate = sf.read(path)
dataset = [data, data]
x_train = np.array(dataset)
y_train = keras.utils.to_categorical([0, 1], num_classes=2)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)