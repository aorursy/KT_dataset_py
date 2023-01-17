# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load necessary packages

from keras.datasets import cifar10

from keras.utils import np_utils

from matplotlib import pyplot as plt

import numpy as np

from PIL import Image
# load the data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# Lets determine the dataset characteristics

print('Training Images: {}'.format(X_train.shape))

print('Testing Images: {}'.format(X_test.shape))
# Now for a single image 

print(X_train[0].shape)
# create a grid of 3x3 images

for i in range(0,9):

    plt.subplot(330 + 1 + i)

    img = X_train[i]

    plt.imshow(img)

    

# show the plot

plt.show()
# Building a convolutional neural network for object recognition on CIFAR-10



# fix random seed for reproducibility

seed = 6

np.random.seed(seed) 



# load the data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()



# normalize the inputs from 0-255 to 0.0-1.0

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train = X_train / 255.0

X_test = X_test / 255.0
# class labels shape

print(y_train.shape)

print(y_train[0])
# hot encode outputs

Y_train = np_utils.to_categorical(y_train)

Y_test = np_utils.to_categorical(y_test)

num_classes = Y_test.shape[1]



print(Y_train.shape)

print(Y_train[0])
# start building the model - import necessary layers

from keras.models import Sequential

from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D

from keras.optimizers import SGD



def allcnn(weights=None):

    # define model type - Sequential

    model = Sequential()



    # add model layers - Convolution2D, Activation, Dropout

    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))

    model.add(Activation('relu'))

    model.add(Conv2D(96, (3, 3), padding = 'same'))

    model.add(Activation('relu'))

    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2,2)))

    model.add(Dropout(0.5))



    model.add(Conv2D(192, (3, 3), padding = 'same'))

    model.add(Activation('relu'))

    model.add(Conv2D(192, (3, 3), padding = 'same'))

    model.add(Activation('relu'))

    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2,2)))

    model.add(Dropout(0.5))



    model.add(Conv2D(192, (3, 3), padding = 'same'))

    model.add(Activation('relu'))

    model.add(Conv2D(192, (1, 1), padding = 'valid'))

    model.add(Activation('relu'))

    model.add(Conv2D(10, (1, 1), padding = 'valid'))



    # add GlobalAveragePooling2D layer with Softmax activation

    model.add(GlobalAveragePooling2D())

    model.add(Activation('softmax'))

    

    # load the weights

    if weights:

        model.load_weights(weights)

    

    # return model

    return model



# define hyper parameters



learning_rate = 0.01

weight_decay = 1e-6

momentum = 0.9



# define weights and build model

weights = '../input/weights/all_cnn_weights_0.9088_0.4994.hdf5'

model = allcnn(weights)



# define optimizer and compile model

sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



# print model summary

print (model.summary())



# test the model with pretrained weights

scores = model.evaluate(X_test, Y_test, verbose=1)

print("Accuracy: %.2f%%" % (scores[1]*100))
# make dictionary of class labels and names

classes = range(0,10)



names = ['airplane',

        'automobile',

        'bird',

        'cat',

        'deer',

        'dog',

        'frog',

        'horse',

        'ship',

        'truck']



# zip the names and classes to make a dictionary of class_labels

class_labels = dict(zip(classes, names))



# generate batch of 9 images to predict

batch = X_test[100:109]

labels = np.argmax(Y_test[100:109],axis=-1)



# make predictions

predictions = model.predict(batch, verbose = 1)
# print our predictions

print (predictions)
# these are individual class probabilities, should sum to 1.0 (100%)

for image in predictions:

    print(np.sum(image))
# use np.argmax() to convert class probabilities to class labels

class_result = np.argmax(predictions,axis=-1)

print (class_result)
# create a grid of 3x3 images

fig, axs = plt.subplots(3, 3, figsize = (15, 6))

fig.subplots_adjust(hspace = 1)

axs = axs.flatten()



for i, img in enumerate(batch):



    # determine label for each prediction, set title

    for key, value in class_labels.items():

        if class_result[i] == key:

            title = 'Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])

            axs[i].set_title(title)

            axs[i].axes.get_xaxis().set_visible(False)

            axs[i].axes.get_yaxis().set_visible(False)

            

    # plot the image

    axs[i].imshow(img)

    

# show the plot

plt.show()
y_pred=model.predict(X_test, verbose = 1)

class_result_y_pred = np.argmax(y_pred,axis=-1)
class_result_y_pred
i2c = np.load("../input/inc2int-to-category/datasets_704654_1230900_int2categroy.npy")

i2c
class_result_y_pred = [i2c[i] for i in class_result_y_pred]
class_result_y_pred
idx=np.arange(1,10000+1)

answers = pd.DataFrame({"id":idx,"label":class_result_y_pred})

answers
answers.to_csv("answers.csv",index=0)