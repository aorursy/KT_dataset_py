import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Input, Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



import os

print(os.listdir("../input"))



from matplotlib import pyplot as plt



batch_size = 128

num_classes = 10

epochs = 12
# mnist data

df_train = pd.read_csv('../input/mnist_train.csv')

df_test = pd.read_csv('../input/mnist_test.csv')



#normalization

X_train = df_train.iloc[:, 1:785]

y_train = df_train.iloc[:, 0]

X_train = X_train.values.astype('float32')/255.



X_test = df_test.iloc[:, 1:785]

y_test = df_test.iloc[:, 0]

X_test = X_test.values.astype('float32')/255.



#reshape

output_X_train = X_train.reshape(-1,28,28,1)

output_X_test = X_test.reshape(-1,28,28,1)



#y label one-hot encoding

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



print(X_train.shape, X_test.shape)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(output_X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(output_X_test, y_test))

score = model.evaluate(output_X_test, y_test, verbose=0)
print('Test loss:', score[0])

print('Test accuracy:', score[1])

predictions = model.predict(output_X_test, batch_size=200)
import random

image_idx = random.randint(1,10000)-1

first_image = X_test[image_idx]

pixels = first_image.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()

print(predictions[image_idx].argmax(axis=0))