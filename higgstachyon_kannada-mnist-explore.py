!pip install scikit-plot
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

from matplotlib import pyplot as plt

import gzip

import os



from keras.utils.data_utils import get_file

import numpy as np





def load_data():

    """Loads the Kannada-MNIST dataset.

    # Returns

        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    """

    dirname = os.path.join('datasets', 'Kannada-mnist')

    base = 'https://github.com/vinayprabhu/Kannada_MNIST/blob/master/data/output_tensors/MNIST_format/'

    files = ['y_kannada_MNIST_train-idx1-ubyte.gz', 'X_kannada_MNIST_train-idx3-ubyte.gz',

             'y_kannada_MNIST_test-idx1-ubyte.gz', 'X_kannada_MNIST_test-idx3-ubyte.gz']



    paths = []

    for fname in files:

        paths.append(get_file(fname,

                              origin=base + fname+'?raw=true',

                              cache_subdir=dirname))



    with gzip.open(paths[0], 'rb') as lbpath:

        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)



    with gzip.open(paths[1], 'rb') as imgpath:

        x_train = np.frombuffer(imgpath.read(), np.uint8,

                                offset=16).reshape(len(y_train), 28, 28)



    with gzip.open(paths[2], 'rb') as lbpath:

        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)



    with gzip.open(paths[3], 'rb') as imgpath:

        x_test = np.frombuffer(imgpath.read(), np.uint8,

                               offset=16).reshape(len(y_test), 28, 28)



    return (x_train, y_train), (x_test, y_test)
# Load the Kannada-MNIST dataset:

(x_train, y_train), (x_test, y_test)=load_data()
from matplotlib import pyplot as plt



plt.figure(figsize=(10,4))



for i in range(50):

  plt.subplot(5,10,i+1)

  plt.imshow(x_train[i],cmap='gray')

  plt.axis('Off')

  plt.title(y_train[i])  

# plt.tight_layout()

plt.suptitle('Training set')



plt.figure(figsize=(10,4))

for i in range(50):

  plt.subplot(5,10,i+1)

  plt.imshow(x_test[i],cmap='gray')

  plt.axis('Off')

  plt.title(y_test[i]) 

# plt.tight_layout()

plt.suptitle('Test set')
plt.figure(figsize=(5,2))



for i in range(10):

  plt.subplot(1,10,i+1)

  

  plt.imshow(np.mean(x_train[y_train==i],axis=0),cmap='gray')

  plt.axis('Off')

  plt.title(i)  

# plt.tight_layout()

plt.suptitle('Training set - Classwise mean images')



plt.figure(figsize=(5,2))

for i in range(10):

  plt.subplot(1,10,i+1)

  

  plt.imshow(np.mean(x_test[y_test==i],axis=0),cmap='gray')

  plt.axis('Off')

  plt.title(i)  

# plt.tight_layout()

plt.suptitle('Training set - Classwise mean images')
from sklearn.decomposition import PCA

import scikitplot as skplt

pca = PCA(random_state=1)

pca.fit(x_test.reshape(10000,784))

skplt.decomposition.plot_pca_component_variance(pca)

plt.show()
'''Trains a simple convnet on the dataset.

Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

'''

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



batch_size = 128

num_classes = 10

epochs = 12



# input image dimensions

img_rows, img_cols = 28, 28



if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

Y_train = keras.utils.to_categorical(y_train, num_classes)

Y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(x_train, Y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, Y_test))

score = model.evaluate(x_test, Y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])