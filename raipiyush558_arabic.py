# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

df_train_image = pd.read_csv("../input/csvTrainImages 13440x1024.csv", header = None)
df_train_label = pd.read_csv("../input/csvTrainLabel 13440x1.csv", header = None)
df_test_image = pd.read_csv("../input/csvTestImages 3360x1024.csv", header = None )
df_test_label = pd.read_csv("../input/csvTestLabel 3360x1.csv", header = None )
df_train_image.head()
df_train_label.head()
#changing the value of above dataset into floating type 
#training images
trainImg = df_train_image.values.astype('float32')
#train label 
trainLabel = df_train_label.values.astype('int32') -1 

#test images
testImg = df_test_image.values.astype('float32')

testLabel = df_test_label.values.astype('int32') -1 
trainImg
testImg
from tflearn.data_utils import to_categorical

#One Hot encoding of train labels.
trainLabel = to_categorical(trainLabel,28)

#One Hot encoding of test labels.
testlabel = to_categorical(testLabel,28)
trainLabel
testLabel
# reshape input images to 28x28x1
# this is to do create an input for neural network
trainImg = trainImg/255
testImg = testImg/255
trainImg= trainImg.reshape([-1, 32, 32, 1])
testImg = testImg.reshape([-1, 32, 32, 1])
#featuring mean value for 
import tflearn.data_utils as du
trainImg, mean1 = du.featurewise_zero_center(trainImg)
testImg, mean2 = du.featurewise_zero_center(testImg)
testImg[0]
# Building convolutional network
recognizer = Sequential()

#first layer of cnn
recognizer.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (32,32,1)))
recognizer.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
recognizer.add(MaxPool2D(pool_size=(2,2)))
recognizer.add(Dropout(0.25))

#second layer of cnn
recognizer.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
recognizer.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
recognizer.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
recognizer.add(Dropout(0.25))

#fully connected layer 
recognizer.add(Flatten())
recognizer.add(Dense(units = 512, input_dim = 1024, activation = 'relu'))
recognizer.add(Dense(units = 256, activation = "relu"))
recognizer.add(Dropout(0.5))
recognizer.add(Dense(28, activation = "softmax"))  
recognizer.summary()
trainImg.shape


testImg.shape
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
recognizer.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False,  
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
datagen.fit(trainImg)
recognizer.fit_generator(datagen.flow(trainImg,trainLabel, batch_size=100),
                             epochs = 30, verbose = 2, steps_per_epoch=trainImg.shape[0] // 100)
prediction = recognizer.predict(testImg)
prediction = np.argmax(prediction, axis =1 )
cm = confusion_matrix(testLabel, prediction)
cm
accuracy = sum(cm[i][i] for i in range(28)) / testLabel.shape[0]
print("accuracy = " + str(accuracy))
#if kernel is useful plese upvote
