import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_8 import *
print("Setup Complete")

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

batch_size = 16

fashion_model = Sequential()
fashion_model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model.add(Conv2D(16, (3, 3), activation='relu'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

fashion_model.fit(x, y,
          batch_size=batch_size,
          epochs=3,
          validation_split = 0.2)

fashion_model_1 = Sequential()
fashion_model_1.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_1.add(Conv2D(16, (3, 3), activation='relu', strides=2))
fashion_model_1.add(Flatten())
fashion_model_1.add(Dense(128, activation='relu'))
fashion_model_1.add(Dense(num_classes, activation='softmax'))

fashion_model_1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

fashion_model_1.fit(x, y,
          batch_size=batch_size,
          epochs=3,
          validation_split = 0.2)
q_1.check()
# q_1.solution()
# That's nice but what happened to dropouts?

fashion_model_2 = Sequential()
fashion_model_2.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_2.add ( Dropout (0.7) )
fashion_model_2.add(Conv2D(16, (3, 3), activation='relu', strides=2))
fashion_model_2.add(Flatten())
fashion_model_2.add(Dense(128, activation='relu'))
fashion_model_2.add ( Dropout (0.2) )
fashion_model_2.add(Dense(num_classes, activation='softmax'))

fashion_model_2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

fashion_model_2.fit(x, y,
          batch_size=batch_size,
          epochs=3,
          validation_split = 0.2)
