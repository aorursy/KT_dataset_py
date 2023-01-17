import warnings

warnings.filterwarnings('ignore')

import numpy as np

np.seed = 1324

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import seaborn as sns

sns.set(style="white")

%matplotlib inline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import cv2



import sys

import os



import keras
from keras.models import Sequential

from keras import layers

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import SpatialDropout2D

from keras.layers import Dropout

from keras.layers import Activation, BatchNormalization

from keras.optimizers import SGD

from keras import regularizers



from keras.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler



from keras.utils.np_utils import to_categorical



import tensorflow as tf



from PIL import Image, ImageEnhance 



from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

# from keras.datasets import cifar10
def submit_file(y_pred, filename):

    submit = pd.Series(y_pred.reshape(y_pred.shape[0]), name='target').replace({0: 'Bird', 1: 'Airplane'})

    submit.to_csv(filename +'.csv', index_label='id', header=True)

    print('file created')
#Функция для визуализации изображений

def viz_img(image, y=None):

    if y != None:

        plt.title(['Bird', 'Airplane'][int(y)])

    plt.imshow(image)

    plt.show()
def flip_image(array, show=False):

    '''for horizontal flip the picture'''

    image_obj = Image.fromarray(array.astype('uint8'))

    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)

    flipped_array = np.array(rotated_image, dtype=np.int)

    if show:

        plt.imshow(flipped_array)

    return flipped_array



def contrast_image(array, c_level=4.0, show=False):

    '''for contrast modification of the picture'''

    image_obj = Image.fromarray(array.astype('uint8'))

    enhancer = ImageEnhance.Contrast(image_obj)

    enhanced_im = enhancer.enhance(c_level)

    

    contrasted_array = np.array(enhanced_im, dtype=np.int)

    if show:

        plt.imshow(contrasted_array)

    return contrasted_array
train_x = pd.read_csv('../input/train_x.csv', index_col=0, header=None)

train_y = pd.read_csv('../input/train_y.csv', index_col=0)

test_x = pd.read_csv('../input/test_x.csv', index_col=0, header=None)
#3 слоя размером 32х32 они "вытянуты" в вектор 

train_x.shape, test_x.shape, train_y.shape
train_y.head(3)
train_y.replace(['Bird', 'Airplane'], [0,1], inplace=True)



X = train_x.values.reshape(7200, 32, 32, 3)

X_test = test_x.values.reshape(4800, 32, 32, 3)

y_train = train_y.values



# x_test_padded = []

# for img in x_test:

#     img = np.pad(img,((2,2),(2,2),(0,0)), 'constant')

#     x_test_padded.append(img)

# x_test_padded = np.stack(x_test_padded)



# x_train_padded = []

# for img in x_train:

#     img = np.pad(img,((2,2),(2,2),(0,0)), 'constant')

#     x_train_padded.append(img)

# x_train_padded = np.stack(x_train_padded)
viz_img(X[100], y_train[100])
viz_img(X[452], y_train[452])
# Old Functions

def create_flip_set(X):

    my_flipped_imgs = []

    for img in X:

        nimg = flip_image(img)

        my_flipped_imgs.append(nimg)

    return np.stack(my_flipped_imgs)



def create_contrast_set(X):

    my_contrasted_imgs = []

    for img in X:

        nimg = contrast_image(img)

        my_contrasted_imgs.append(nimg)

    return np.stack(my_contrasted_imgs)
# callbacks = [

#     TerminateOnNaN(), ModelCheckpoint('bird_plane_lenet.hdf5', verbose=1, monitor='val_loss', save_best_only=True),

# #    LearningRateScheduler(step_decay)

# ]
def lr_schedule(epoch):

    lrate = 0.001

    if epoch > 75:

        lrate = 0.0005

    if epoch > 100:

        lrate = 0.0003

    return lrate

 
X = X.astype('float32')

X_test = X_test.astype('float32')

x_train, x_test, y_train, y_test = train_test_split(X, y_train, test_size=0.3, random_state=12000)
 #z-score

mean = np.mean(x_train, axis=(0,1,2,3))

std = np.std(x_train,axis=(0,1,2,3))

x_train = (x_train-mean) / (std + 1e-7)

x_test = (x_test-mean) / (std + 1e-7)

X_test = (X_test-mean) / (std + 1e-7)
weight_decay = 1e-4

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), 

                                            input_shape=x_train.shape[1:]))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

 

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))

 

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Activation('elu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))

 

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

 

model.summary()
#data augmentation

datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    )

datagen.fit(x_train)
#training

batch_size = 64

 

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

model.compile(loss='binary_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=125,

                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
plt.plot(model.history.history['acc'], c='blue', label='train')

plt.plot(model.history.history['val_acc'], c='green', label='val')

plt.legend()

plt.title('Accuracy')
plt.plot(model.history.history['loss'], c='blue', label='train')

plt.plot(model.history.history['val_loss'], c='green', label='val')

plt.legend()

plt.title('Loss')
y_pred = model.predict_classes(X_test)
submit_file(y_pred, 'best')
#save model to disk

model_json = model.to_json()

with open('model.json', 'w') as json_file:

    json_file.write(model_json)

model.save_weights('model.h5') 