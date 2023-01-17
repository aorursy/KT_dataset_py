from keras.models import Model, Sequential

from keras.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape

from keras.callbacks import EarlyStopping

from keras import backend as K

from keras.optimizers import Adam

from tensorflow.metrics import mean_iou

import tensorflow as tf

import numpy as np

import pandas as pd

import glob

import PIL

from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline



from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from warnings import filterwarnings

filterwarnings('ignore')
import re

numbers = re.compile(r'(\d+)')

def numericalSort(value):

    parts = numbers.split(value)

    parts[1::2] = map(int, parts[1::2])

    return parts
filelist_trainx = sorted(glob.glob('../input/*/trainx/*.bmp'), key=numericalSort)

X_train = np.array([np.array(Image.open(fname)) for fname in filelist_trainx])



filelist_trainy = sorted(glob.glob('../input/*/trainy/*.bmp'), key=numericalSort)

Y_train = np.array([np.array(Image.open(fname)) for fname in filelist_trainy])
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.25, random_state = 101)
plt.figure(figsize=(20,9))

plt.subplot(2,4,1)

plt.imshow(X_train[0])

plt.subplot(2,4,2)

plt.imshow(X_train[3])

plt.subplot(2,4,3)

plt.imshow(X_train[54])

plt.subplot(2,4,4)

plt.imshow(X_train[77])

plt.subplot(2,4,5)

plt.imshow(X_train[100])

plt.subplot(2,4,6)

plt.imshow(X_train[125])

plt.subplot(2,4,7)

plt.imshow(X_train[130])

plt.subplot(2,4,8)

plt.imshow(X_train[149])

plt.show()
plt.figure(figsize=(20,9))

plt.subplot(2,4,1)

plt.imshow(Y_train[0], cmap = plt.cm.binary_r)

plt.subplot(2,4,2)

plt.imshow(Y_train[3], cmap = plt.cm.binary_r)

plt.subplot(2,4,3)

plt.imshow(Y_train[54], cmap = plt.cm.binary_r)

plt.subplot(2,4,4)

plt.imshow(Y_train[77], cmap = plt.cm.binary_r)

plt.subplot(2,4,5)

plt.imshow(Y_train[100], cmap = plt.cm.binary_r)

plt.subplot(2,4,6)

plt.imshow(Y_train[125], cmap = plt.cm.binary_r)

plt.subplot(2,4,7)

plt.imshow(Y_train[130], cmap = plt.cm.binary_r)

plt.subplot(2,4,8)

plt.imshow(Y_train[149], cmap = plt.cm.binary_r)

plt.show()
def iou(y_true, y_pred, smooth = 100):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return jac
def dice_coef(y_true, y_pred, smooth = 100):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def model_seg(epochs_num,savename):



    img_input = Input(shape= (192, 256, 3))

    # Block 1

    x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)

    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='conv2',strides= (1,1))(x)

    x = Activation('relu')(x)

    

    # Block 2

    x = Conv2D(128, (3, 3), padding='same', name='conv3',strides= (1,1))(x)

    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='conv4',strides= (1,1))(x)

    x = Activation('relu')(x)



    # Block 3

    x = Conv2D(256, (3, 3), padding='same', name='conv5',strides= (1,1))(x)

    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='conv6',strides= (1,1))(x)

    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='conv7',strides= (1,1))(x)

    x = Activation('relu')(x)



    # Block 4

    x = Conv2D(512, (3, 3), padding='same', name='conv8',strides= (1,1))(x)

    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv9',strides= (1,1))(x)

    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv10',strides= (1,1))(x)

    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv11',strides= (1,1))(x)

    x = Activation('relu')(x)

    

    # Block 5

    x = Conv2D(512, (3, 3), padding='same', name='conv12',strides= (1,1))(x)

    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='conv13',strides= (1,1))(x)

    x = Activation('relu')(x)

    x = Conv2D(4096, (7, 7), padding='same', name='conv14',strides= (1,1))(x)

    x = Dropout(0.5)(x)

    

    

    # Block 6

    x = Conv2D(4096, (1, 1), padding='same', name='conv15',strides= (1,1))(x)

    x = Dropout(0.5)(x)

    x = Activation('relu')(x)

    x = Conv2D(1, (1, 1), padding='same', name='conv16',strides= (1,1))(x)

    x = Activation('softmax')(x)

    



    pred = Reshape((192, 256))(x)

    

    model = Model(inputs=img_input, outputs=pred)

    

    model.compile(optimizer= Adam(lr = 0.003), loss= ['binary_crossentropy'], metrics=[iou, dice_coef])

    model.summary()

    hist = model.fit(x_train, y_train, epochs= epochs_num, batch_size= 1, verbose=1)



    model.save(savename)
model_seg(epochs_num= 1, savename= 'model_epochs_1.h5')
img_input = Input(shape= (192, 256, 3))

# Block 1

x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)

x = Activation('relu')(x)

x = Conv2D(64, (3, 3), padding='same', name='conv2',strides= (1,1))(x)

x = Activation('relu')(x)



# Block 2

x = Conv2D(128, (3, 3), padding='same', name='conv3',strides= (1,1))(x)

x = Activation('relu')(x)

x = Conv2D(128, (3, 3), padding='same', name='conv4',strides= (1,1))(x)

x = Activation('relu')(x)



# Block 3

x = Conv2D(256, (3, 3), padding='same', name='conv5',strides= (1,1))(x)

x = Activation('relu')(x)

x = Conv2D(256, (3, 3), padding='same', name='conv6',strides= (1,1))(x)

x = Activation('relu')(x)

x = Conv2D(256, (3, 3), padding='same', name='conv7',strides= (1,1))(x)

x = Activation('relu')(x)



# Block 4

x = Conv2D(512, (3, 3), padding='same', name='conv8',strides= (1,1))(x)

x = Activation('relu')(x)

x = Conv2D(512, (3, 3), padding='same', name='conv9',strides= (1,1))(x)

x = Activation('relu')(x)

x = Conv2D(512, (3, 3), padding='same', name='conv10',strides= (1,1))(x)

x = Activation('relu')(x)

x = Conv2D(512, (3, 3), padding='same', name='conv11',strides= (1,1))(x)

x = Activation('relu')(x)



# Block 5

x = Conv2D(512, (3, 3), padding='same', name='conv12',strides= (1,1))(x)

x = Activation('relu')(x)

x = Conv2D(512, (3, 3), padding='same', name='conv13',strides= (1,1))(x)

x = Activation('relu')(x)

x = Conv2D(4096, (7, 7), padding='same', name='conv14',strides= (1,1))(x)

x = Dropout(0.5)(x)





# Block 6

x = Conv2D(4096, (1, 1), padding='same', name='conv15',strides= (1,1))(x)

x = Dropout(0.5)(x)

x = Activation('relu')(x)

x = Conv2D(1, (1, 1), padding='same', name='conv16',strides= (1,1))(x)

x = Activation('softmax')(x)





pred = Reshape((192, 256))(x)

model_0 = Model(inputs=img_input, outputs=pred)

model_0.compile(optimizer= Adam(lr = 0.003), loss= ['binary_crossentropy'], metrics=[iou, dice_coef])

model_0.load_weights('model_epochs_1.h5')
print('~~~~~~~~~~~~~~~~~~~~~~~~Stats after 1 epoch~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')



print('\n----------------------------On Train Set -----------------------------------\n')

res = model_0.evaluate(x_train, y_train, batch_size= 1)

print('IOU: {:.2f}'.format(res[1]*100))

print("Dice Coefficient: {:.2f}".format(res[2]*100))



print('\n-----------------------------On Test Set ------------------------------------\n')

res = model_0.evaluate(x_test, y_test, batch_size= 1)

print('IOU: {:.2f}'.format(res[1]*100))

print("Dice Coefficient: {:.2f}".format(res[2]*100))