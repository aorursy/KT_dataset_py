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
import numpy as np

import keras

from keras.layers import Dense,Conv2D,BatchNormalization,Activation,DepthwiseConv2D,MaxPooling2D

from keras.layers import Dropout,Flatten,Input

from keras.utils import to_categorical

from keras.layers.merge import add

from sklearn.model_selection import train_test_split

from keras.models import Model,Sequential

from keras.callbacks import EarlyStopping,ModelCheckpoint

from keras.initializers import RandomUniform

import cv2

%matplotlib inline

import matplotlib.pyplot as plt
test_images = np.load('../input/Test_images.npy')

test_labels = np.load('../input/Test_labels.npy')

Train_images = np.load('../input/Train_images.npy')

Train_labels = np.load('../input/Train_labels.npy')
Train_images.shape
Train_images,vald_images,Train_labels,vald_labels = train_test_split(Train_images,Train_labels,test_size = 0.3,random_state = 33)
print("Validation set : " + str(vald_images.shape) + " labels : "+ str(vald_labels.shape))

print("test set : "+ str(test_images.shape) + " labels : "+ str(test_labels.shape))

print("Training set : " + str(Train_images.shape) + " labels : "+ str(Train_labels.shape))
Train_labels = to_categorical(Train_labels)

test_labels = to_categorical(test_labels)

vald_labels = to_categorical(vald_labels)
Train_images = [cv2.resize(img, (129,129)) for img in Train_images[:,:,:,:]]
vald_images = [cv2.resize(img,(129,129)) for img in vald_images[:,:,:,:]]
test_images = [cv2.resize(img,(129,129)) for img in test_images[:,:,:,:]]
Train_images = np.array(Train_images)

vald_images = np.array(vald_images)

test_images = np.array(test_images)
test_images.shape
Train_images.shape
# hyperparameters

stride = 1

CHANNEL_AXIS = 3
earlystopiing  = EarlyStopping(monitor = 'val_loss',patience = 3 )

filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath,monitor = 'acc')



callback_list = [checkpoint]
def res_block(y,filters,pooling = False,dropout = 0.0):

    shortcut_path = y

    y = DepthwiseConv2D(depth_multiplier=2, kernel_size = (3,3), strides = stride,padding = 'same')(y)

    y = BatchNormalization(axis = CHANNEL_AXIS)(y)

    y = Activation('relu')(y)

    y = DepthwiseConv2D(depth_multiplier=2, kernel_size = (3,3), strides = stride,padding = 'same')(y)

    y = BatchNormalization(axis = CHANNEL_AXIS)(y)

    y = Activation('relu')(y)

    #add shortcut to residual path

    

    shortcut_path = add([y,Conv2D(filters = filters,kernel_size = (3,3),strides = stride,padding = "same")(shortcut_path)])

    

    if pooling:

        shortcut_path = MaxPooling2D((2,2))(shortcut_path)

    if dropout != 0.0:

        shortcut_path = Dropout(dropout)(shortcut_path)

    y  = BatchNormalization(axis = CHANNEL_AXIS)(shortcut_path)

    y  = Activation('relu')(y)

    

    return y



def non_resblock(y,filters1,dropout = 0.0):

    compressed = int(filters1/2)

    y = Conv2D(filters = compressed,kernel_size = (1,1),strides = 1,padding = 'same')(y)

    y = Activation('relu')(y)

   # y = MaxPooling2D(pool_size=(2, 2),strides = 1,padding = 'valid')(y)

    y = Conv2D(filters = compressed,kernel_size = (3,3),strides = 1,padding = 'same')(y)

    y = Activation('relu')(y)

    y = Conv2D(filters = filters1,kernel_size = (1,1),strides = 1,padding = 'same')(y)

    y = MaxPooling2D(pool_size=(2, 2),strides = (2,2),padding = 'valid')(y)

    

    return y







 

    

    

    

    
inp = Input(shape=(129,129,3))

x = inp

x = Conv2D(32,(3,3),strides = stride,padding = "valid")(x)

x = BatchNormalization(axis = CHANNEL_AXIS)(x)

x = Activation("relu")(x)



#stack residual and non_residual layers

x = res_block(x,filters=128,dropout = 0.2)

x = non_resblock(x,32,dropout = 0.2)

x = res_block(x,filters=128,dropout = 0.2,pooling = True)

x = non_resblock(x,32,dropout = 0.2)

x = res_block(x,filters=128,dropout = 0.2,pooling = True)

x = non_resblock(x,32,dropout = 0.2)

x = res_block(x,filters=128,dropout = 0.2)

x = Flatten()(x)

x = Dense(128,activation = "relu")(x)

x = Dropout(0.3)(x)

x = Dense(64,activation = "relu")(x)

x = Dropout(0.2)(x)

x = Dense(32,activation = "relu")(x)

x = Dropout(0.2)(x)

x = Dense(10,activation = "softmax")(x)



hybrid_resnet = Model(inp,x,name = 'hybrid_resnet')

hybrid_resnet.summary()

hybrid_resnet.compile(optimizer='adam',loss = "categorical_crossentropy",metrics = ["accuracy"])
history = hybrid_resnet.fit(Train_images,Train_labels,epochs = 40 ,batch_size = 100,callbacks = callback_list,validation_data = (vald_images,vald_labels))
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'vald'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'vald'], loc='upper left')

plt.show()
hybrid_resnet.evaluate(test_images,test_labels,batch_size = 100)