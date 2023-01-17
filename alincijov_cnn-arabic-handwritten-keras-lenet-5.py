import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

from tensorflow.keras.optimizers import SGD
train_images = pd.read_csv('../input/ahcd1/csvTrainImages 13440x1024.csv')

train_label = pd.read_csv('../input/ahcd1/csvTrainLabel 13440x1.csv')



test_images = pd.read_csv('../input/ahcd1/csvTestImages 3360x1024.csv')

test_label = pd.read_csv('../input/ahcd1/csvTestLabel 3360x1.csv')
X_train = train_images.to_numpy()

y_train = train_label.to_numpy()



X_test = test_images.to_numpy()

y_test = test_label.to_numpy()
nr_classes = len(np.unique(train_label))
X_train = X_train.reshape(13439, 32, 32, 1)

y_train = tf.keras.backend.one_hot(y_train, nr_classes)

y_train = y_train.numpy().reshape(13439, 28)



X_test = X_test.reshape(3359, 32, 32, 1)

y_test = tf.keras.backend.one_hot(y_test, nr_classes)

y_test = y_test.numpy().reshape(3359, 28)
# Instantiate an empty sequential model

model = Sequential()

# C1 Convolutional Layer

model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'tanh',

input_shape = (32,32,1), padding = 'same'))

 

# S2 Pooling Layer

model.add(AveragePooling2D(pool_size = 2, strides = 2, padding = 'valid'))

 

# C3 Convolutional Layer

model.add(Conv2D(filters = 16, kernel_size = 5, strides = 1,activation = 'tanh',

padding = 'valid'))

# S4 Pooling Layer

model.add(AveragePooling2D(pool_size = 2, strides = 2, padding = 'valid'))

 

# C5 Convolutional Layer

model.add(Conv2D(filters = 120, kernel_size = 5, strides = 1,activation = 'tanh',

padding = 'valid'))

 

# Flatten the CNN output to feed it with fully connected layers

model.add(Flatten())

 

# FC6 Fully Connected Layer

model.add(Dense(units = 84, activation = 'tanh'))

 

# FC7 Output layer with softmax activation

model.add(Dense(units = 28, activation = 'softmax'))

 

# print the model summary

model.summary()
def lr_schedule(epoch):

    # initiate the learning rate with value = 0.0005

    lr = 5e-4

    # lr = 0.0005 for the first two epochs, 0.0002 for the next three epochs,

    # 0.00005 for the next four, then 0.00001 thereafter.

    if epoch > 2:

        lr = 2e-4

    elif epoch > 5:

        lr = 5e-5

    elif epoch > 9:

        lr = 1e-5

    return lr
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr_schedule(0)), metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=124, epochs=500,

validation_data=(X_test, y_test), verbose=0, shuffle=True)
plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()