import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import matplotlib.image as mping

import seaborn as sns

sns.set(style = 'white', context = 'notebook', palette = 'deep')

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Lambda

from keras.optimizers import RMSprop, SGD, Adagrad, Adam 

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from matplotlib import pyplot

from math import pi

from math import cos

from math import floor

from keras.callbacks import Callback

from keras import backend

from numpy import argmax

from subprocess import check_output

from keras.layers import Convolution2D, MaxPooling2D

import keras

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



## Let's make sure our data is loaded 

print(check_output(["ls", "../input"]).decode("utf8"))
## Let's see what our training data looks like.

print("Training Shape:", train.shape)

train.head()

## Let's see what our testing data looks like.



print("Testing Shape: ", test.shape)

test.head()

## Variable for our pixel data

trainX = (train.iloc[:,1:].values).astype('float32')



## Variable for our targets digits. 

trainY = train.iloc[:,0].values.astype("int32")



## Variable for our testing pixels

testX = test.values.astype('float32')



## For the convinience of our Random Access Memory, let's give him some space.

del train 
## Let's see what our variables looks like now

trainX
## Here's our digits category that our model needs to be good at recognizing. 

trainY
## Let's visualize the count of each value



Y = train["label"]



Y.value_counts()



plot = sns.countplot(Y)

## Let's visualize some digits



trainX = trainX.reshape(trainX.shape[0], 28, 28)



for i in range(6,9):

    plt.subplot(330 + (i+1))

    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))

    plt.title(trainY[i]);
## Reshape Training

trainX = trainX.reshape(-1,28,28,1)

trainX.shape



## Reshape Testing

testX = testX.reshape(-1,28,28,1)

testX.shape
## Standardization



meanX = trainX.mean().astype(np.float32)

std_X = trainX.std().astype(np.float32)



def standardization(x):

    return (x-meanX)/std_X

## One Hot Encoding



trainY = to_categorical(trainY, num_classes = 10)

classes = trainY.shape[1]

classes
## Data Split



trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.10, random_state=2)

trainX.shape
testX.shape
trainY.shape
testY.shape

g = plt.imshow(trainX[0][:,:,0])
## For the love of data, let's create some more!



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(trainX)
## Architechture



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

## Model parameters



epochs = 25 ## We keep it low for minimum training time. Increase if better performance is needed.

batch_size = 86

verbose = 2

step_per_epoch = trainX.shape[0] // batch_size

loss = 'categorical_crossentropy'

learning_rate = 0.1

Momentum_opti = SGD(lr= learning_rate, momentum = 0.9, nesterov = False)
Nesterov_opti = SGD(lr = learning_rate, momentum = 0.9, nesterov = True)
Adagrad_opti = Adagrad(lr = learning_rate, epsilon = None, decay = 0.9)
RMSProp_opti = RMSprop(lr = learning_rate, rho = 0.9, decay = 0.9, epsilon = 1e-10)
Adam_opti = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-10, decay = 0.9, amsgrad = False)
## Change optimizer variable in order to test training with another optimizer.

model.compile(optimizer = Nesterov_opti, loss = loss, metrics = ["accuracy"])
## Training Time!



history = model.fit_generator(datagen.flow(trainX, trainY, batch_size = batch_size), epochs = epochs, validation_data = (testX, testY), verbose = verbose,

                             steps_per_epoch = step_per_epoch)

                             
_, train_acc = model.evaluate(trainX, trainY, verbose = 0)
_, test_acc = model.evaluate(testX, testY, verbose = 0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
predictions = model.predict_classes(testX, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})



submissions.to_csv("DR.csv", index=False, header=True)