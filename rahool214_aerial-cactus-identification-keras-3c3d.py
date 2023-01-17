# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from keras.preprocessing.image import load_img, img_to_array



import keras

from keras.layers import Dense, Conv2D, MaxPooling2D,Flatten, Dropout, Activation, Input

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#### Load training Data CSV

trainDF = pd.read_csv('../input/train.csv')



#### Get training images 

trainImgList = list(trainDF['id'])

trainImg = []



for img in trainImgList:

    originalImage = load_img(f'../input/train/train/{img}')

    arrayImage = img_to_array(originalImage) / 255

    trainImg.append(arrayImage)

    

trainImgNP = np.array(trainImg)



#### Get training Lables

trainLabels = trainDF['has_cactus'].values
#### Model Helper Functions

def getConvLayer(inputLayer, kernelSize = 2 ,filters = 64):

    conv1 = Conv2D(filters, kernelSize, activation = 'relu')(inputLayer)

    #conv2 = Conv2D(filters, kernelSize, activation = 'relu')(conv1)

    pool1 = MaxPooling2D()(conv1)

    return pool1



def getFlattenLayer(inputLayer):

    flat = Flatten()(inputLayer)

    return flat



def getDenseLayer(inputLayer, units = 512, rate = .5):

    dense1 = Dense(units, activation = 'relu')(inputLayer)

    drop1 = Dropout(rate)(dense1)

    return drop1



def getOutLayer(inputLayer):

    out1 = Dense(1, activation = 'sigmoid')(inputLayer)

    return out1



def getModel():

    #### Create Model and show Summary

    inputLayer = Input(shape = [32, 32, 3])



    # 1st ConvLayer

    conv1 = getConvLayer(inputLayer)



    # 2nd ConvLayer

    conv2 = getConvLayer(conv1, filters = 32)

    

    # 3rd ConvLayer

    conv3 = getConvLayer(conv2, filters = 16)



    # FlattenLayer

    flat = getFlattenLayer(conv3)



    # 1st DenseLayer

    dense1 = getDenseLayer(flat)



    # 2nd DenseLayer

    dense2 = getDenseLayer(dense1)



    # 3rd DenseLayer

    dense3 = getDenseLayer(dense2)



    # OutputLayer

    out = getOutLayer(dense3)



    # Create Model

    model = Model(inputLayer, out)

    #model.summary()

    

    return model



#### Learn Rate reduction

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
#### Compile and train Model

model = getModel()



model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])



model.fit(x = trainImgNP, y = trainLabels, batch_size = 64, epochs = 100, validation_split = .1, callbacks=[learning_rate_reduction])
#### Get testing images 

testImgList = os.listdir("../input/test/test")

testImg = []



for img in testImgList:

    originalImage = load_img(f'../input/test/test/{img}')

    arrayImage = img_to_array(originalImage) / 255

    testImg.append(arrayImage)

    

testImgNP = np.array(testImg)
#### get Predictions for test Data

testPred = model.predict(testImgNP)

testPredRound = np.round(testPred)

subPred = [int(x) for [x] in testPredRound]
#### Create a Submission DF

sub = pd.DataFrame({'id' : testImgList, 'has_cactus' : subPred})

sub.to_csv('submission_3C_3D.csv', index = False)