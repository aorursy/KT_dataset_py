import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data, train_size=50000, val_size=5000)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

fashion_model_1 = Sequential()
# Specify the rest of the model
fashion_model_1.add(Conv2D(12, kernel_size=(3, 3), strides=2,
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_1.add(Conv2D(12, (3, 3), strides=2, activation='relu'))
fashion_model_1.add(Flatten())
fashion_model_1.add(Dense(128, activation='relu'))
fashion_model_1.add(Dense(num_classes, activation='softmax'))

# Compile fashion_model_1

fashion_model_1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Fit fashion_model_1

fashion_model_1.fit(x, y,
          batch_size=128,
          epochs=5,
          validation_split = 0.2)
# Your code for fashion_model_2 below

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

fashion_model_2 = Sequential()
# Specify the rest of the model
fashion_model_2.add(Conv2D(24, kernel_size=(3, 3), strides=2,
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_2.add(Conv2D(24, (3, 3), strides=2, activation='relu'))
fashion_model_2.add(Conv2D(24, (3, 3), activation='relu'))
fashion_model_2.add(Flatten())
fashion_model_2.add(Dense(128, activation='relu'))
fashion_model_2.add(Dense(num_classes, activation='softmax'))

# Compile fashion_model_1

fashion_model_2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Fit fashion_model_1

fashion_model_2.fit(x, y,
          batch_size=128,
          epochs=5,
          validation_split = 0.2)
# Your code for fashion_model_3 below

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

fashion_model_3 = Sequential()
# Specify the rest of the model
fashion_model_3.add(Conv2D(24, kernel_size=(3, 3), strides=2,
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
fashion_model_3.add(Dropout(0.05))
fashion_model_3.add(Conv2D(24, (3, 3), strides=2, activation='relu'))
fashion_model_3.add(Dropout(0.05))
fashion_model_3.add(Conv2D(24, (3, 3), activation='relu'))
fashion_model_3.add(Dropout(0.05))
fashion_model_3.add(Flatten())
fashion_model_3.add(Dense(128, activation='relu'))
fashion_model_3.add(Dense(num_classes, activation='softmax'))

# Compile fashion_model_1

fashion_model_3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# Fit fashion_model_1

fashion_model_3.fit(x, y,
          batch_size=128,
          epochs=5,
          validation_split = 0.2)