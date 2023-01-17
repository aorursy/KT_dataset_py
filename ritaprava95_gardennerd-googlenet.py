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
import numpy as np

import pandas as pd

import tensorflow as tf

import keras.backend as K

from keras.utils import to_categorical 

from keras.models import Model

from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout, Dense, Flatten, Activation, concatenate

from keras.initializers import glorot_uniform, he_uniform, Zeros

import matplotlib.pyplot as plt
train_X = np.load("/kaggle/input/train-data/train_X.npy")



train_y = np.asarray(pd.read_csv("/kaggle/input/train-data/train.csv").iloc[:,1])



train_y = to_categorical(train_y)
kernel_init = glorot_uniform()

bias_init = Zeros()

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):

    conv_1x1 = Conv2D(filters_1x1, kernel_size=(1,1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer = bias_init)(x)

    

    conv_3x3 = Conv2D(filters_3x3_reduce, kernel_size=(1,1), padding='same', activation ='relu', kernel_initializer=kernel_init, bias_initializer = bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3, kernel_size=(3,3), padding='same', activation ='relu', kernel_initializer=kernel_init, bias_initializer = bias_init)(x)

    

    conv_5x5 = Conv2D(filters_5x5_reduce, kernel_size=(1,1), padding='same', activation ='relu', kernel_initializer=kernel_init, bias_initializer = bias_init)(x)

    conv_5x5 = Conv2D(filters_5x5, kernel_size=(3,3), padding='same', activation ='relu', kernel_initializer=kernel_init, bias_initializer = bias_init)(x)

    

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)

    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init)(pool_proj)

    

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    

    return output

    
input_layer = Input(shape=(100, 100, 3))



x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init)(input_layer)

x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)

x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)

x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)

x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)





x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32, name='inception_3a')



x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64, name='inception_3b')



x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)



x = inception_module(x, filters_1x1=192, filters_3x3_reduce=96,filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64, name='inception_4a')





x1 = AveragePooling2D((5, 5), strides=3)(x)

x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)

x1 = Flatten()(x1)

x1 = Dense(1024, activation='relu')(x1)

x1 = Dropout(0.7)(x1)

x1 = Dense(103, activation='softmax', name='auxilliary_output_1')(x1)



x = inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64, name='inception_4b')



x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64, name='inception_4c')



x = inception_module(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32,filters_5x5=64, filters_pool_proj=64, name='inception_4d')





x2 = AveragePooling2D((5, 5), strides=3)(x)

x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)

x2 = Flatten()(x2)

x2 = Dense(1024, activation='relu')(x2)

x2 = Dropout(0.7)(x2)

x2 = Dense(103, activation='softmax', name='auxilliary_output_2')(x2)



x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, name='inception_4e')



x = MaxPooling2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)



x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, name='inception_5a')



x = inception_module(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128, name='inception_5b')



x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(103, activation='softmax', name='output')(x)

model = Model(input_layer, [x1, x2, x], name='inception_v1')



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
model.fit(train_X, [train_y, train_y, train_y], validation_split=0.2, epochs = 100, batch_size=64)
plt.plot(model.history.history['val_auxilliary_output_1_accuracy'])

plt.plot(model.history.history['val_auxilliary_output_2_accuracy'])

plt.plot(model.history.history['val_output_accuracy'])

plt.plot(model.history.history['auxilliary_output_1_accuracy'])

plt.plot(model.history.history['auxilliary_output_2_accuracy'])

plt.plot(model.history.history['output_accuracy'])

plt.legend(['validation accuracy 1','validation accuracy 2', 'validation accuracy 3', 'training accuracy 1','training accuracy 2', 'training accuracy 3'])

plt.ylabel('Accuracy')

plt.xlabel('Epochs')