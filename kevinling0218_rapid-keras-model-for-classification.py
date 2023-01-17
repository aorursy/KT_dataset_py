# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Import necessary modules and libraries

import keras

from sklearn.model_selection import train_test_split

from keras.layers import Input,Dense,Activation,BatchNormalization,Flatten,Conv2D

from keras.layers import AveragePooling2D,MaxPooling2D

from keras.models import Model
# Load training data

train= pd.read_json('../input/train.json')



#Create 3 bands for CNN processing. 

band_1 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_1']])

band_2 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_2']])

train_data = np.concatenate([band_1[:,:,:,np.newaxis],band_2[:,:,:,np.newaxis],((band_1+band_2)/2)[:,:,:,np.newaxis]], axis=-1)



label=train['is_iceberg']

X_train, X_cv, y_train, y_cv = train_test_split(train_data,label, test_size=0.20)
def model(input_shape):

    X_input = Input(input_shape)



    #ConV - BN -Relu Block

    X = Conv2D(32, kernel_size=(3,3), name='conv0')(X_input)

    X = BatchNormalization(axis=3, name='bn0')(X)

    X = Activation('relu')(X)



    #MaxPool

    X=MaxPooling2D((2,2),name='max_pool0')(X)



    #Flatten and fully connected

    X = Flatten()(X)

    X = Dense(1, activation='sigmoid', name='fc0')(X)

    

    #Create keras model instance

    model = Model(input=X_input, outputs=X, name='TestModel')



    return model

# Create the model

TestModel=model((75,75,3))



# Compile the model to configure learning process

TestModel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])



# Train the model using training data

TestModel.fit(x=X_train, y=y_train, epochs=100, batch_size=50)