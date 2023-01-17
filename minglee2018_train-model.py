# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import sys

import os

import cv2

import numpy as np

import copy

import random

from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image

import keras

import tensorflow as tf

from keras import layers

from keras import optimizers

from keras import regularizers

from keras import backend as K

from keras.optimizers import adam 



from keras.models import Sequential

from keras.models import Model, load_model



import seaborn as sns 



from keras import models

from keras import optimizers

from keras import callbacks

from keras import losses

from keras import regularizers

from keras.layers import Dense, Dropout, Flatten 

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D

from keras.layers import Activation, Dense

from keras import optimizers 





from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot



# from tensorflow.keras.models import BatchNormalization

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten, Input

from keras.layers import Convolution2D, MaxPooling2D,ZeroPadding2D

from keras.optimizers import SGD

import theano



from tensorflow_core.python.keras.utils.data_utils import Sequence

from keras.layers  import BatchNormalization



from sklearn.decomposition import PCA

from tensorflow.keras.optimizers import SGD

import math

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import LearningRateScheduler

img_fps_train  = '/kaggle/input/train'

img_fps_test   = '/kaggle/input/test'

img_fps_validation = '/kaggle/input/valid'

def wfile_data(root): 

    fp = []

    for path, subdirs, file in os.walk(root): 

        for name in file: 

            fp.append(os.path.join(path,name))

    return sorted(fp)
X_train = wfile_data(img_fps_train)

X_test  = wfile_data(img_fps_test)

X_valid = wfile_data(img_fps_validation)

Y_train = []

Y_test = []

Y_valid = []
for path in X_train: 

    Y_train.append(int(path.split("/")[-2])-1)

for path in X_test: 

    Y_test.append(int(path.split("/")[-2])-1)

for path in X_valid: 

    Y_valid.append(int(path.split("/")[-2])-1)

len(X_train),len(Y_train),len(X_test),len(Y_test),len(X_valid),len(Y_valid)


class FlowerClassifyGenerator(keras.utils.Sequence):

    def __init__(self, img_fps, labels, batch_size=8,

                 img_size=(224, 224), n_channels=3,

                 no_classes=10, shuffle=True, mode='arc'):



        self.img_size = img_size

        self.batch_size = batch_size

        self.mode = mode

        

        self.img_fps = img_fps

        self.labels = labels

        assert len(self.img_fps) == len(self.labels)

        self.ids = range(len(self.img_fps))



        self.n_channels = n_channels

        self.no_classes = no_classes

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        return int(np.floor(len(self.ids) / self.batch_size))



    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        temp_ids = [self.ids[k] for k in indexes]

        X, y = self.__data_generation(temp_ids)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.ids))

        if self.shuffle:

            np.random.shuffle(self.indexes)

    def load_img(self, img_fp):

        try:

            img = cv2.imread(img_fp)[:, :, ::-1]

        except TypeError:

            return None

        return img

     

    def __data_generation(self, ids):

        # Initialization

        X = np.empty((0, *self.img_size, self.n_channels), dtype=np.float32)

        y = np.empty((0,*self.img_size))



        for index, id_ in enumerate(ids):

            img_fp = self.img_fps[id_]

            img = self.load_img(img_fp)

            label = self.labels[id_]

            if img is None:

                continue

            img = cv2.resize(img,self.img_size)

            img = img.astype(np.float32)

            img = img / 255

            img = np.expand_dims(img, axis=0)  

            X = np.vstack((X, img))

            y = np.append(y,label)

        y = keras.utils.to_categorical(y, num_classes=n_classes)

        y = y.astype(np.float32)        

        return X, y
input_shape_flower = (224,224,3)

batch_size = 8 

epochs = 10

n_classes = 102

n_channels = 3

size_sample = len(X_train)
params = {'batch_size':batch_size,

          'img_size':(224,224),

          'n_channels':3,

          'no_classes':102, 

          'shuffle': True}

Generator_train = FlowerClassifyGenerator(X_train,Y_train,**params)

Generator_valid = FlowerClassifyGenerator(X_valid,Y_valid,**params)
plt.imshow(Generator_train[0][0][0])
from keras.regularizers import Regularizer
model = Sequential()



model.add(Conv2D(16,(3,3),activation='relu', padding = 'same',input_shape = (224,224,3)))

model.add(BatchNormalization())

model.add(Conv2D(16,(3,3),activation='relu',padding = 'same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(32,(3,3),activation = 'relu', padding = 'same'))

model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),activation ='relu', padding = 'same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size= (2,2))) 



model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), activation ='relu', padding = 'same'))

model.add(MaxPooling2D(pool_size = (2,2)))



# model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))

# model.add(BatchNormalization())

# model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())

model.add(Dense(512,kernel_regularizer=regularizers.l2(0.01)))



model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(102, activation='softmax'))



model.summary()
from keras import metrics

print( " == Compiling Model ==")

sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov= True)

model.compile(optimizer = sgd, 

                loss = 'categorical_crossentropy',

                metrics=['accuracy'])



reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.1,

                                  patience = 2, min_lr=0.00001)



# model.compile(loss='mean_squared_error',

#               optimizer='sgd',

#               metrics=[metrics.mae, metrics.categorical_accuracy])
history = model.fit_generator(generator=Generator_train,

                    steps_per_epoch= math.floor(len(Generator_train.ids)//batch_size),

                    epochs= 25, validation_data=Generator_valid,

                    validation_steps = math.floor(len(Generator_valid.ids)//batch_size), 

                    verbose = 1,callbacks=[reduce_lr], shuffle=True)


# from numpy import array

# from keras.models import Sequential

# from keras.layers import Dense

# from matplotlib import pyplot

# # prepare sequence

# X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# # create model

# model = Sequential()

# model.add(Dense(2, input_dim=1))

# model.add(Dense(1))

# model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

# # train model

# history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)

# # plot metrics

# pyplot.plot(history.history['mean_squared_error'])

# pyplot.plot(history.history['mean_absolute_error'])

# pyplot.plot(history.history['mean_absolute_percentage_error'])

# pyplot.plot(history.history['cosine_proximity'])

# pyplot.show()
model.save_weights("train_model.h5")

model.load_weights("train_model.h5")

model.summary()
history.history.keys()
leaning_rate = history.history['lr']

leaning_rate
def train_model_show(_history):

    ## plot some result

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 10))



    # plot train and val loss

    ax1.plot(_history.history['loss'], label = 'Train_loss')

    ax1.plot(_history.history['val_loss'], label = 'Val_loss')

    ax1.set_title('Train_loss and Val_loss')

    ax1.set_ylabel('loss')

    ax1.set_label('epochs')

    ax1.legend()



    # plot train and val accuracy

    ax2.plot(_history.history['accuracy'], label = 'train accuracy')

    ax2.plot(_history.history['val_accuracy'], label = 'val_accuracy')

    ax2.set_title('train accuracy and validation accuracy')

    ax2.set_ylabel('acc')

    ax2.set_label('epochs')

    ax2.legend()

    plt.show()

    

#     fig, (ax3) = plt.subplots(1, 1, figsize = (15, 10))

#     ax3.plot(_history.history['categorical_accuracy'], label = 'categorical_accuracy')

#     ax3.plot(_history.history['val_categorical_accuracy'], label = 'val_categorical_accuracy')

#     ax3.set_title('categorical_accuracy and val_categorical_accuracy ')

#     ax3.set_ylabel('acc')

#     ax3.set_label('epochs')

#     ax3.legend()

#     plt.show()

train_model_show(history) 
input_shape = (224,224)

def Preprocessing_Image(img_fps_path):

#     try:

    img = cv2.imread(img_fps_path)[:,:,::-1]

#     except TypeError: 

#         return None

    img = cv2.resize(img,(224,224))

    img = img.astype(np.float32)

    img = img / 255

    img = np.expand_dims(img, axis=0)  

    return img 
arr = np.empty( 0, dtype = np.float32 )

new_arr = keras.utils.to_categorical(arr, num_classes=102)



for fps_path in X_test: 

    new_img  = model.predict(Preprocessing_Image(fps_path))

    new_arr = np.vstack((new_arr,new_img))

    
y_true=np.argmax(new_arr, axis=-1)

flat_array = np.asarray(Y_test)
print(y_true.shape)

print(flat_array.shape) 
from sklearn.metrics import confusion_matrix



# Set the figure size

plt.figure(figsize=(100, 50))



# Calculate the confusion matrix

cm = confusion_matrix(y_true, y_pred=flat_array)



# Normalize the confusion matrix

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0



# Visualize the confusion matrix

sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True);