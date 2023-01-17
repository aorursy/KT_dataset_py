import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

from tensorflow.keras.models import Sequential

from IPython.display import FileLink



DATA_DIR = '../input/follow-path-in-minecraft/training_data.npy'

FILE_NAME = 'training_data.npy'

HEIGHT = 600

WIDTH = 800

FINAL_HEIGHT = 40

FINAL_WIDTH = 80

EPOCHS = 8

BATCH_SIZE = 16
data = np.load(DATA_DIR, allow_pickle=True)
X = np.array([data[i][0] for i in range(len(data))])

y = np.array([data[i][1] for i in range(len(data))])



X = X.reshape([-1, FINAL_HEIGHT, FINAL_WIDTH, 1])



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = Sequential()



model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(FINAL_HEIGHT, FINAL_WIDTH, 1)))

model.add(MaxPool2D(pool_size=(2, 2), strides=2))



model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(3, 3), strides=2))



model.add(Flatten())

model.add(Dense(2048, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2048, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(3, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[x_test, y_test])
model.save('my_model.h5')

FileLink('my_model.h5')