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
data_train_file="../input/digit-recognizer/train.csv"

data_test_file="../input/digit-recognizer/test.csv"

train_set=pd.read_csv(data_train_file)

test_set=pd.read_csv(data_test_file)
import tensorflow as tf

import keras as keras

import numpy as np

import os

from sys import getsizeof

#from random import sample

import math

import random



import statistics

import glob



from sklearn.preprocessing import Normalizer

from sklearn.metrics import confusion_matrix,classification_report



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import pandas as pd



from keras.models import Sequential,Model

from keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Input, BatchNormalization,LeakyReLU

from keras.layers.convolutional import Conv2D,MaxPooling2D, AveragePooling2D

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.vis_utils import plot_model

from keras import optimizers
#Own implementation with inception-like layers.

def model_creation(size,channels,output_in):

    freq16_in = Input(shape=(size,size,channels))





    #batch16_1 = BatchNormalization()(freq16_in)



    conv16_1 = Conv2D(32, kernel_size=(5,5), activation="relu",padding='same')(freq16_in)

    max16_1 = MaxPooling2D(pool_size=(3,3), strides=(2, 2), padding='same')(conv16_1)

    batch16_2 = BatchNormalization()(max16_1)



    conv16_2 = Conv2D(64, kernel_size=(5,5), activation="relu",padding='same')(batch16_2)

    max16_2 = MaxPooling2D(pool_size=(3,3), strides=(2, 2), padding='same')(conv16_2)

    batch16_3 = BatchNormalization()(max16_2)



    conv16_3 = Conv2D(128, kernel_size=(3,3), activation="relu",padding='same')(batch16_3)

    max16_3 = MaxPooling2D(pool_size=(3,3), strides=(2, 2), padding='same')(conv16_3)

    batch16_4 = BatchNormalization()(max16_3)



    towerOne16_1 = Conv2D(32, kernel_size=(1,1), activation="relu",padding='same')(batch16_4)

    towerTwo16_1 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(batch16_4)

    towerThree16_1 = Conv2D(32, kernel_size=(5,5), activation='relu', padding='same')(batch16_4)

    maxinc16_1 = MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same')(batch16_4)

    towerfour16_1 = Conv2D(32, kernel_size=(1,1), activation="relu",padding='same')(maxinc16_1)

    inc1 = concatenate([towerOne16_1, towerTwo16_1, towerThree16_1, towerfour16_1], axis=3)



    batch16_5 = BatchNormalization()(inc1)



    towerOne = Conv2D(32, kernel_size=(1,1), activation="relu",padding='same')(batch16_5)

    towerTwo = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(batch16_5)

    towerThree = Conv2D(32, kernel_size=(5,5), activation='relu', padding='same')(batch16_5)

    maxinc16_2 = MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same')(batch16_5)

    towerfour16_2 = Conv2D(32, kernel_size=(1,1), activation="relu",padding='same')(maxinc16_2)

    inc2 = concatenate([towerOne, towerTwo, towerThree, towerfour16_2], axis=3)





    max16_4 = MaxPooling2D(pool_size=(3,3), strides=(2, 2), padding='same')(inc2)





    batch16_6 = BatchNormalization()(max16_4)



    towerOne = Conv2D(128, kernel_size=(1,1), activation="relu",padding='same')(batch16_6)

    towerTwo = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(batch16_6)

    towerThree = Conv2D(128, kernel_size=(5,5), activation='relu', padding='same')(batch16_6)

    maxinc16_3 = MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same')(batch16_6)

    towerfour16_3 = Conv2D(128, kernel_size=(1,1), activation="relu",padding='same')(maxinc16_3)

    inc3 = concatenate([towerOne, towerTwo, towerThree, towerfour16_3], axis=3)





    batch16_7 = BatchNormalization()(inc3)





    towerOne = Conv2D(128, kernel_size=(1,1), activation="relu",padding='same')(batch16_7)

    towerTwo = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(batch16_7)

    towerThree = Conv2D(128, kernel_size=(5,5), activation='relu', padding='same')(batch16_7)

    maxinc16_4 = MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='same')(batch16_7)

    towerfour16_4 = Conv2D(128, kernel_size=(1,1), activation="relu",padding='same')(maxinc16_4)

    inc4 = concatenate([towerOne, towerTwo, towerThree, towerfour16_4], axis=3)





    avpool = AveragePooling2D(pool_size=(2,2))(inc4)



    flat = Flatten()(avpool)



    dense1 = Dense(512, activation='relu')(flat)

    drop1 = Dropout(0.4)(dense1)



    #dense2 = Dense(128, activation='relu')(drop1)

    #drop2 = Dropout(0.5)(dense2)





    output = Dense(output_in, activation='softmax')(drop1)







    modelname = Model(inputs = freq16_in, outputs=output)

    return modelname
#Split into train and validation

from sklearn.model_selection import train_test_split



train_data ,val_data = train_test_split(train_set,test_size=0.1)
#Separate into labels and pixels

def df_to_np(data_frame):

    return np.asarray(data_frame).reshape((data_frame.shape[0],int(np.sqrt(data_frame.shape[1])),int(np.sqrt(data_frame.shape[1])),1))   



train_x=df_to_np(train_data.drop("label",axis=1))

train_y=pd.get_dummies(train_data["label"])



val_x=df_to_np(val_data.drop("label",axis=1))

val_y=pd.get_dummies(val_data["label"])



test_x=df_to_np(test_set)
#create model

model1=model_creation(channels=1,size=28,output_in=10) #Create model

model1.compile(optimizer="Adam",

              loss="categorical_crossentropy", metrics=["accuracy"])
#Train model

model1.fit(train_x, train_y, epochs=5, batch_size=128,validation_data=(val_x,val_y))

model1.compile(optimizer=keras.optimizers.Adam(lr=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])

model1.fit(train_x, train_y, epochs=3, batch_size=128,validation_data=(val_x,val_y))

model1.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

model1.fit(train_x, train_y, epochs=2, batch_size=128,validation_data=(val_x,val_y))

model1.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss="categorical_crossentropy", metrics=["accuracy"])

model1.fit(train_x, train_y, epochs=1, batch_size=128,validation_data=(val_x,val_y))
#Predict

test_y=np.argmax(model1.predict(test_x),axis=1)

submission = pd.DataFrame({ 'ImageId': range(1,test_y.shape[0]+1),

                            'Label': test_y })

submission.to_csv("submission.csv", index=False)

import os

os.chdir(r'/kaggle/working')



from IPython.display import FileLink

FileLink(r'submission.csv')