# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/digit-recognizer/train.csv')

test=pd.read_csv('../input/digit-recognizer/test.csv')

print(train.head())
#null-check

print(train.isnull().any().describe())

print(train.shape)
X_train = train.drop('label', axis=1)

y_train = train.iloc[:, 0]



# normalize the pixel intensity values.

X_train = X_train / 255.0

test = test / 255.0



# reshape to a matrix of m * 28 * 28 where 1 is for 1D.

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

from keras.utils import np_utils 

Y_train =  np_utils.to_categorical(y_train, num_classes = 10)



# split set into training and validation. 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
# Building the CNN



# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense, Dropout

# Initializing layers

model = Sequential()

# Adding a convolutional layer

model.add(Convolution2D(filters = 32, kernel_size = (5,5),

                 activation ='relu', input_shape = (28,28,1)))

model.add(Convolution2D(filters = 32, kernel_size = (5,5), 

                 activation ='relu'))

# Adding a pooling layer

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



# Adding a 2nd convolutional layer

model.add(Convolution2D(filters = 64, kernel_size = (3,3),

                 activation ='relu'))

model.add(Convolution2D(filters = 64, kernel_size = (3,3), 

                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

#Dropout reduces overfitting



# Flattening

model.add(Flatten())



# Making full connection

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



# Compiling the CNN

model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



# Image Augmentation, creating more images with ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator

imagegen = ImageDataGenerator(

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



imagegen.fit(X_train)
# fit the model with generated images. 

fitobj = model.fit_generator(imagegen.flow(X_train,y_train, batch_size=86),

                              epochs = 1, validation_data = (X_test,y_test),

                              verbose = 1, steps_per_epoch=X_train.shape[0])
# predict results

results = model.predict(test)



# change results in appropriate format for the submission

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)