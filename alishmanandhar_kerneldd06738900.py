import numpy as np

import keras
from keras.models import Sequential
from keras.optimizers import RMSprop,Adam,SGD,Adamax
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras.layers import Dense, Flatten

from keras.datasets import mnist
from matplotlib import pyplot as plt
%matplotlib inline
batch_size = 600
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
x_1 = x_train[0]
x_1 = np.reshape(x_1, (28,28))
plt.imshow(x_1, cmap='Greys')
y_train[0]
x_train.shape

        
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train /= 255
x_test /= 255
print("X Train Shape: ", x_train.shape)
print("X Test Shape: ", x_test.shape)

x_train = np.expand_dims(x_train,axis=3)
x_test = np.expand_dims(x_test,axis=3)

y_train[0]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train[0]
print("Train Labels Shape: ", y_train.shape)
print("Test Labels Shape: ", y_test.shape)
y_train[0]
model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3), 
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(128, kernel_size=(3, 3), 
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3, 3), 
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), 
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=(3, 3), 
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(keras.layers.Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax', input_shape=(784,)))

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=15, batch_size=batch_size,validation_data=(x_test,y_test))
x_pred = x_test[20]
x_pred = np.expand_dims(x_pred, axis=0)
x_pred.shape
model.predict_classes(x_pred)
y_test[20]
evaluate = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss: ", evaluate[0])
print("Test Accuracy: ", evaluate[1])

model.save('mnist.h5')

!ls
