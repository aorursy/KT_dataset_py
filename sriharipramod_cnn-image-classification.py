# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%env CUDA_DEVICE_ORDER=PCI_BUS_ID

%env CUDA_VISIBLE_DEVICES=1
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
import warnings

warnings.filterwarnings('ignore')



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from keras.utils import to_categorical

from keras.datasets import cifar10



from PIL import Image



from keras.models import Model, Sequential

from keras.layers import Flatten, Dense, Dropout

from keras.layers import Convolution2D, MaxPooling2D

from keras.layers import BatchNormalization, GlobalAveragePooling2D

from keras.utils import to_categorical

from keras.optimizers import Adam





# Get Inception architecture from keras.applications

from keras.applications.inception_v3 import InceptionV3
def load_cifar10(resize=False):

    train = np.load('../input/cifar-traintest/cifar_train.npz')

    x_train = train['data']

    y_train = train['labels']

  

    test = np.load('../input/cifar-traintest/cifar_test.npz')

    x_test = test['data']

    y_test = test['labels']

    

    if resize:

        x_train=resize_all(x_train, resize)

        x_test=resize_all(x_test, resize)

    

    x_train = x_train.astype('float32')/255.

    x_test = x_test.astype('float32')/255.

    

    return(x_train, y_train, x_test, y_test)





def resize(p, size):

    return Image.fromarray(p).resize(size=(size,size))



def resize_all(arr, size):

    t = []

    for i in range(arr.shape[0]):

        t.append(np.array(resize(arr[i], size)))

        

#     t = np.array(t, dtype='float32')

#     t /= 255.



    return(np.array(t))
batch_size = 64

nb_classes = 10



img_rows, img_cols = 32, 32    # input image dimensions

img_channels = 3               # The CIFAR10 images are RGB.
x_train, train_labels, x_test, test_labels = load_cifar10()





y_train = to_categorical(train_labels, nb_classes)

y_test = to_categorical(test_labels, nb_classes)
plt.figure(figsize=(20,10))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(x_train[i])

    plt.axis('off')
def custom_convnet(nb_classes, learn_rate, inp_shape):

    model = Sequential()

    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', # valid

                            input_shape=inp_shape, 

                            activation='relu'))

    model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', 

                            activation='relu'))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax'))

    

    adam = Adam(lr=learn_rate)

    model.compile(loss='categorical_crossentropy',

                  metrics=['accuracy'],optimizer=adam)

    

    return(model)

model = custom_convnet(nb_classes=10, learn_rate=0.001, 

                       inp_shape=(img_rows,img_cols,img_channels))

model.fit(x_train[:50000], y_train[:50000],

          validation_data=(x_test, y_test),

          batch_size=batch_size,

          epochs=4)



model.save('cifar10_savedmodel.h5')          # Saves the weights along with the graph

# model.save_weights('cifar10_savedmodel.h5')  # Saves the weights only
model = custom_convnet(nb_classes=10, learn_rate=0.002, inp_shape=(img_rows,img_cols,img_channels))

model.fit(x_train[:5000], y_train[:5000],

          validation_data=(x_test, y_test),

          batch_size=batch_size,

          epochs=15)
(x_train, train_labels), (x_test, test_labels) = cifar10.load_data()

size=224

x_train=resize_all(x_train, size)

x_test=resize_all(x_test, size)



x_train = x_train.astype('float32')/255.

x_test = x_test.astype('float32')/255.



y_train = to_categorical(train_labels, nb_classes)

y_test = to_categorical(test_labels, nb_classes)
def inception_tl(nb_classes, freez_wts=True, learn_rate=0.001):

    trained_model = InceptionV3(include_top=False,weights='imagenet')

    x = trained_model.output

    x = GlobalAveragePooling2D()(x)

    pred_inception= Dense(nb_classes,activation='softmax')(x)

    model = Model(inputs=trained_model.input,outputs=pred_inception)

    

    for layer in trained_model.layers:

        layer.trainable=(1-freez_wts)

    

    adam = Adam(lr=learn_rate)

    model.compile(loss='categorical_crossentropy',

                  metrics=['accuracy'],optimizer=adam)

    

    return(model)
model = inception_tl(nb_classes=nb_classes, freez_wts=True)

model.fit(x_train[:5000], y_train[:5000],

          validation_data=(x_test, y_test),

          batch_size=batch_size,

          epochs=5)