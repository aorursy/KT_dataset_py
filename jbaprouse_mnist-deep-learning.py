# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(3)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

sns.set(style='white', context='notebook', palette='deep')



y_train = train["label"]

x_train = train.drop(labels="label", axis=1)

del train

# plot on a graph the label counts

pl = sns.countplot(y_train)
# now we check if the data is clean

# look for null values

x_train.isnull().values.any()

y_train.isnull().values.any()
# normalize the data to go between [0,1] not [0,255]

x_train = x_train / 255.0

test = test / 255.0

# now we reshape the pixels from 1x784 to 28x28x1 1dim to 3dim (height, width, canal) and the -1(all) is the number of rows

x_train = x_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes = 10)
# split the training and test sets 90:10

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
x_train.shape
from keras.optimizers import SGD

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



model.fit(x_train, y_train,

          epochs=20,

          batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
# now adjust our data for differences

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



datagen.fit(x_train)
# store model trained without data changes

old_model = model



# now we fit the model

model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=128), steps_per_epoch=x_train.shape[0], 

                    epochs=1, validation_data=(x_test, y_test), verbose=2)
# predict results

results = model.predict(test)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)