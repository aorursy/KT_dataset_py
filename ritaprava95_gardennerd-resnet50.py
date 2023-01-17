# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import tensorflow as tf

import keras.backend as K

from keras.utils import to_categorical 

from keras.models import Model

from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D, BatchNormalization, Dropout, Dense, Flatten, Activation, Add, concatenate

from keras.initializers import glorot_uniform, he_uniform, Zeros

import matplotlib.pyplot as plt

train_X = np.load("/kaggle/input/train-data/train_X.npy")



train_y = np.asarray(pd.read_csv("/kaggle/input/train-data/train.csv").iloc[:,1])



train_y = to_categorical(train_y)
kernel_init = glorot_uniform()



def identity_block(X, F1, F2, F3, k, stage, block):

    conv_name_base = 'Conv'+str(stage)+str(block)

    bn_name_base = 'BN'+str(stage)+str(block)

    

    X_shortcut = X # I will need this to add with convolutional o/p

    

    X = Conv2D(F1, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=kernel_init, name=conv_name_base+'2a')(X)

    X = BatchNormalization(axis=3, name=bn_name_base+'_a')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(F2, kernel_size=(k,k), strides=(1,1), padding='same', kernel_initializer=kernel_init, name=conv_name_base+'2b')(X)

    X = BatchNormalization(axis=3, name=bn_name_base+'_b')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=kernel_init, name=conv_name_base+'2c')(X)

    X = BatchNormalization(axis=3, name=bn_name_base+'_c')(X)

    

    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

    

    return X
def convolutional_block(X, F1, F2, F3, k, s, stage, block):

    conv_name_base = 'Conv'+str(stage)+str(block)

    bn_name_base = 'BN'+str(stage)+str(block)

    

    X_shortcut = X # I will need this to add with convolutional o/p

    X_shortcut = Conv2D(F3, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_initializer=kernel_init, name=conv_name_base+'1')(X_shortcut)

    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    

    X = Conv2D(F1, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_initializer=kernel_init, name=conv_name_base+'2a')(X)

    X = BatchNormalization(axis=3, name=bn_name_base+'_a')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(F2, kernel_size=(k,k), strides=(1,1), padding='same', kernel_initializer=kernel_init, name=conv_name_base+'2b')(X)

    X = BatchNormalization(axis=3, name=bn_name_base+'_b')(X)

    X = Activation('relu')(X)

    

    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=kernel_init, name=conv_name_base+'2c')(X)

    X = BatchNormalization(axis=3, name=bn_name_base+'_c')(X) 

    

    

    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

    

    return X
X_input = Input(shape=(100,100,3))



    

# Zero-Padding

X = ZeroPadding2D((3, 3))(X_input)

    

# Stage 1

X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)

X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

X = Activation('relu')(X)

X = MaxPooling2D((3, 3), strides=(2, 2))(X)



# Stage 2

X = convolutional_block(X, k= 3, F1=64, F2=64, F3=256, stage = 2, block='a', s = 1)

X = identity_block(X, k=3, F1=64, F2=64, F3=256, stage=2, block='b')

X = identity_block(X, k=3, F1=64, F2=64, F3=256, stage=2, block='c')



# Stage 3

X = convolutional_block(X, k=3, F1=128, F2=128, F3=512, stage=3, block='a', s=2)

X = identity_block(X, k=3, F1=128, F2=128, F3=512, stage=3, block='b')

X = identity_block(X, k=3, F1=128, F2=128, F3=512, stage=3, block='c')

X = identity_block(X, k=3, F1=128, F2=128, F3=512, stage=3, block='d')



# Stage 4

X = convolutional_block(X, k=3, F1=256, F2=256, F3=1024, stage=4, block='a', s=2)

X = identity_block(X, k=3, F1=256, F2=256, F3=1024, stage=4, block='b')

X = identity_block(X, k=3, F1=256, F2=256, F3=1024, stage=4, block='c')

X = identity_block(X, k=3, F1=256, F2=256, F3=1024, stage=4, block='d')

X = identity_block(X, k=3, F1=256, F2=256, F3=1024, stage=4, block='e')

X = identity_block(X, k=3, F1=256, F2=256, F3=1024, stage=4, block='f')



# Stage 5

X = convolutional_block(X, k=3, F1=512, F2=512, F3=2048, stage=5, block='a', s=2)

X = identity_block(X, k=3, F1=512, F2=512, F3=2048, stage=5, block='b')

X = identity_block(X, k=3, F1=512, F2=512, F3=2048, stage=5, block='c')



# AVGPOOL

X = AveragePooling2D(pool_size=(2,2), padding='same')(X)



# Output layer

X = Flatten()(X)

X = Dense(103, activation='softmax', name='fc' + str(103), kernel_initializer = glorot_uniform(seed=0))(X)

    

    

    
# Create model

model = Model(inputs = X_input, outputs = X, name='ResNet50')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_y, validation_split=0.2, epochs = 100, batch_size=128)
import matplotlib.pyplot as plt 

print(model.history.history.keys())

plt.figure(1)

plt.plot(model.history.history['accuracy'])

plt.plot(model.history.history['val_accuracy'])

plt.legend(['training accuracy', 'validation accuracy'])

plt.ylabel('Accuracy')

plt.xlabel('Epochs')