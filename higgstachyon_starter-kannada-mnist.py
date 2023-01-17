!pip install scikit-plot
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Source: https://github.com/dccastro/Morpho-MNIST/blob/bb01283636a79b8b4752ea241c2d263f45c84409/morphomnist/io.py    

import gzip

import struct





def _load_uint8(f):

    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]

    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))

    buffer_length = int(np.prod(shape))

    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)

    return data

	

def load_idx(path: str) -> np.ndarray:

    """Reads an array in IDX format from disk.

    Parameters

    ----------

    path : str

        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns

    -------

    np.ndarray

        Output array of dtype ``uint8``.

    References

    ----------

    http://yann.lecun.com/exdb/mnist/

    """

    open_fcn = gzip.open if path.endswith('.gz') else open

    with open_fcn(path, 'rb') as f:

        return _load_uint8(f)









def load_Kannada_mnist(file_type='npz',ds_dir='../input/kannada_mnist_datataset_paper/Kannada_MNIST_datataset_paper'):

    if file_type=='npz':

        npz_dir=os.path.join(ds_dir,'Kannada_MNIST_npz')  

        y_test=np.load(os.path.join(npz_dir,'Kannada_MNIST','y_kannada_MNIST_test.npz'))

        y_test=y_test.f.arr_0

        X_test=np.load(os.path.join(npz_dir,'Kannada_MNIST','X_kannada_MNIST_test.npz'))

        X_test=X_test.f.arr_0

        y_train=np.load(os.path.join(npz_dir,'Kannada_MNIST','y_kannada_MNIST_train.npz'))

        y_train=y_train.f.arr_0

        X_train=np.load(os.path.join(npz_dir,'Kannada_MNIST','X_kannada_MNIST_train.npz'))

        X_train=X_train.f.arr_0

    else:

        gz_dir=os.path.join(ds_dir,'Kannada_MNIST_Ubyte_gz') 

        y_test=load_idx(os.path.join(gz_dir,'Kannada_MNIST','y_kannada_MNIST_test-idx1-ubyte.gz'))

        X_test=load_idx(os.path.join(gz_dir,'Kannada_MNIST','X_kannada_MNIST_test-idx3-ubyte.gz'))

        y_train=load_idx(os.path.join(gz_dir,'Kannada_MNIST','y_kannada_MNIST_train-idx1-ubyte.gz'))

        X_train=load_idx(os.path.join(gz_dir,'Kannada_MNIST','X_kannada_MNIST_train-idx3-ubyte.gz'))

    return (X_train, y_train), (X_test, y_test)

def load_Dig_mnist(file_type='npz',ds_dir='../input/kannada_mnist_datataset_paper/Kannada_MNIST_datataset_paper'):

    if file_type=='npz':

        npz_dir=os.path.join(ds_dir,'Kannada_MNIST_npz')  

        y_dig=np.load(os.path.join(npz_dir,'Dig_MNIST','y_dig_MNIST.npz'))

        y_dig=y_dig.f.arr_0

        X_dig=np.load(os.path.join(npz_dir,'Dig_MNIST','X_dig_MNIST.npz'))

        X_dig=X_dig.f.arr_0



    else:

        gz_dir=os.path.join(ds_dir,'Kannada_MNIST_Ubyte_gz') 

        y_dig=load_idx(os.path.join(gz_dir,'Dig_MNIST','y_dig_MNIST-idx1-ubyte.gz'))

        X_dig=load_idx(os.path.join(gz_dir,'Dig_MNIST','X_dig_MNIST-idx3-ubyte.gz'))

    return (X_dig, y_dig)

ds_dir='../input/kannada_mnist_datataset_paper/Kannada_MNIST_datataset_paper'

print(os.listdir(ds_dir))

npz_dir=os.path.join(ds_dir,'Kannada_MNIST_npz')

gz_dir=os.path.join(ds_dir,'Kannada_MNIST_Ubyte_gz')

os.listdir(npz_dir),os.listdir(gz_dir)
(x_train, y_train), (x_test, y_test)=load_Kannada_mnist('npz')

##################################################

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

#########################

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

plt.suptitle('Testing set - Classwise mean images')
gz_dir=os.path.join(ds_dir,'Kannada_MNIST_npz') 

os.listdir(os.path.join(gz_dir,'Dig_MNIST'))
(x_train, y_train), (x_test, y_test)=load_Kannada_mnist('gz')

##################################################

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

plt.suptitle('TESTing set - Classwise mean images')
(x_dig_np, y_dig_np)=load_Dig_mnist('npz')

(x_dig_gz, y_dig_gz)=load_Dig_mnist('gz')

plt.figure(figsize=(5,2))



for i in range(10):

  plt.subplot(1,10,i+1)

  

  plt.imshow(np.mean(x_dig_gz[y_dig_gz==i],axis=0),cmap='gray')

  plt.axis('Off')

  plt.title(i)  

# plt.tight_layout()

plt.suptitle('Dig-MNIST(gz) - Classwise mean images')



plt.figure(figsize=(5,2))

for i in range(10):

  plt.subplot(1,10,i+1)

  

  plt.imshow(np.mean(x_dig_np[y_dig_np==i],axis=0),cmap='gray')

  plt.axis('Off')

  plt.title(i)  

plt.suptitle('Dig-MNIST(npz) - Classwise mean images')
from sklearn.decomposition import PCA

import scikitplot as skplt

pca = PCA(random_state=1)

pca.fit(x_test.reshape(10000,784))

skplt.decomposition.plot_pca_component_variance(pca)

plt.show()
# The Dig-MNIST dataset

pca = PCA(random_state=1)

pca.fit(x_dig_np.reshape(10240,784))

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
y_pred=model.predict_classes(x_test)

skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)

plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
y_dig_pred=model.predict_classes(x_dig_np.reshape(x_dig_np.shape[0], img_rows, img_cols, 1))

skplt.metrics.plot_confusion_matrix(y_dig_np, y_dig_pred, normalize=True)

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_dig_np, y_dig_pred))