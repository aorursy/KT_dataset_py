from __future__ import print_function

import numpy as np # linear algebra

np.random.seed(42)

import pandas as pd



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras import backend as K





import matplotlib.pyplot as plt

import matplotlib.cm as cm

%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.head()
data.shape
images = data.iloc[:,1:].values

images = images.astype(np.float)



images = np.multiply(images, 1.0/255.0)

images.shape
def displayImg(img):

    get_img = img.reshape(28,28)

    plt.imshow(get_img, cmap=cm.binary)

    

displayImg(images[10])
labels = data.iloc[:,0]

labels[10]
X_train = images[:40000]

Y_train = labels[:40000]



x_test = images[40000:]

y_test = labels[40000:]



X_train.shape , x_test.shape
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)
Y_train = np_utils.to_categorical(Y_train, 10)

y_test = np_utils.to_categorical(y_test, 10)
X_train.shape , Y_train.shape
model = Sequential()



model.add(Convolution2D(32, 3, 3,

                        border_mode='valid',

                        input_shape=input_shape))

model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adadelta',

              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, nb_epoch=5,

          verbose=1, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test, verbose=1)