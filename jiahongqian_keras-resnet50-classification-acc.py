# !git clone https://github.com/keras-team/keras.git

# !cd ../working/keras

# !python ../working/keras/setup.py install
# !conda install keras -y
!ls /opt/conda/lib/python3.6/site-packages/keras/applications
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd

import random

import os

import matplotlib.pyplot as plt

import keras

from keras.layers import *

from keras.models import *

from keras import layers

from keras.utils.data_utils import get_file

from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam,SGD

from keras import optimizers

from keras.applications.resnet50 import ResNet50

# from keras.applications.resnext import ResNeXt50,ResNeXt101

from keras.applications.densenet import DenseNet169,DenseNet201

from keras.applications.nasnet import NASNetLarge

from keras.layers import Input

from sklearn.model_selection import train_test_split

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/breakhist_dataset/BreakHist_Dataset"))



# Any results you write to the current directory are saved as output.
afiles = os.listdir("../input/breakhist_dataset/BreakHist_Dataset")
Bimgs, Mimgs=[],[]

for file in [afiles[0]]:

    pics= [os.path.join(root,name) for root,dirs,files in os.walk("../input/breakhist_dataset/BreakHist_Dataset/{}/Benign".format(file)) for name in files]

    Bimgs.extend(pics)

    pics = [os.path.join(root,name) for root,dirs,files in os.walk("../input/breakhist_dataset/BreakHist_Dataset/{}/Malignant".format(file)) for name in files]

    Mimgs.extend(pics)

    del pics
fig,ax = plt.subplots(3,3)

for i in range(3):

    for j in range(3):

        img = plt.imread(Bimgs[j+3*i])

        ax[i,j].imshow(img)

        ax[i,j].axis('off')

fig.suptitle('Benign')

###

fig,ax = plt.subplots(3,3)

for i in range(3):

    for j in range(3):

        img = plt.imread(Mimgs[j+3*i])

        ax[i,j].imshow(img)

        ax[i,j].axis('off')

fig.suptitle('Malignant')

plt.show()


shape = cv2.imread(Bimgs[0]).shape

# B_X=np.empty((1,*shape))

# M_X=np.empty((1,*shape))

# for x in Bimgs:

#     im = np.expand_dims(cv2.imread(x),axis=0)

#     B_X=np.append(B_X,im)

#     del im

#resize 因为不同的图像的大小不一样，之后nparray后只能变为一维数组

size = (200,200)

B_X=[]

for x in Bimgs: 

    im = cv2.imread(x)

    im = cv2.resize(im,size,interpolation=cv2.INTER_AREA)

    B_X.append(im)

    del im



M_X=[]

for x in Mimgs: 

    im = cv2.imread(x)

    im = cv2.resize(im,size,interpolation=cv2.INTER_AREA)

    M_X.append(im)

    del im

#列表解析式比较耗内存？

# B_X = [cv2.imread(x) for x in Bimgs]

# M_X = [cv2.imread(y) for y in Mimgs if cv2.imread(y).shape==shape]

B_Y = [0 for x in range(len(Bimgs))]

M_Y = [1 for x in range(len(Mimgs))]

X = np.array(B_X+M_X)

Y = np.array(B_Y+M_Y)
# X_all=np.concatenate((np.array(B_X),np.array(M_X)),axis=0)

# Y_all=np.concatenate((np.array(B_Y),np.array(M_Y)),axis=0)


# X_all = B_X+M_X

# Y_all = B_Y+M_Y

# print(len(X_all),len(B_Y),len(M_Y))

# del B_X,M_X

# print(len(X_all),len(B_Y),len(M_Y))
X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=43)


datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True)  # randomly flip images
batch_size = 32

num_classes = 2

epochs = 100

input_shape = X[0].shape

e = 2


# model = Sequential()

# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape,strides=e))

# model.add(Conv2D(64, (3, 3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

# model.add(Flatten())

# model.add(Dense(128, activation='relu'))

# model.add(Dropout(0.5))

# model.add(Dense(num_classes, activation='softmax'))

# input_tensor = Input(input_shape) 

# from keras.applications.densenet import DenseNet169,DenseNet201

# from keras.applications.nasnet import NASNetLarge

MODEL={'ResNet50':ResNet50, \

      'NASNetLarge':NASNetLarge, \

      'DenseNet169':DenseNet169,'DenseNet201':DenseNet201}
name = 'ResNet50'

model =MODEL[name](input_shape=input_shape,weights=None,include_top=True,classes=num_classes)

model.summary()
from keras.utils.np_utils import to_categorical

y_trainCat=to_categorical(Y_train,num_classes=num_classes)

y_testCat=to_categorical(Y_test,num_classes=num_classes)

y_all=to_categorical(Y,num_classes=num_classes)
# import keras

adam = Adam(lr=0.0001)

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=adam,metrics=['accuracy'])

# history = model.fit(X, y_all, epochs=epochs,validation_split=0.2)

if os.path.exists('breakhis_{}.h5'.format(name)):

    model.load_weights('breakhis_{}.h5'.format(name))

history = model.fit_generator(datagen.flow(X_train,y_trainCat, batch_size=batch_size),

                              steps_per_epoch=len(X_train) / 32, 

                              epochs=epochs,validation_data = [X_test, y_testCat])
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.save('breakhis_{}.h5'.format(name))