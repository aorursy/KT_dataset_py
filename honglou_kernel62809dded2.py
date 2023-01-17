# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn import manifold

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.utils.np_utils import to_categorical

import os

print(os.listdir("../input"))

sns.set(style='white', context='notebook', palette='deep')

from keras import backend as K

K.tensorflow_backend._get_available_gpus()

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
y_train=train.label

x_train=train.drop('label',axis=1)
del train
x_train =x_train / 255.0

test = test / 255.0
x_train =x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

X_train, X_val,Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.4, random_state=10)
plt.imshow(x_train[1][:,:,0])
from keras.models import Sequential
model=Sequential()
from keras.layers import Dense, Dropout, Flatten, Conv2D,MaxPooling2D,Lambda,MaxPool2D

from keras.layers.normalization import BatchNormalization
mean = np.mean(X_train)

std = np.std(X_train)



def standardize(x):

    return (x-mean)/std
model = Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))

model.add(Conv2D(32,(3,3),activation="relu"))

model.add(Conv2D(32,(3,3),activation="relu"))

    

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation="relu"))

model.add(Conv2D(64,(3,3),activation="relu"))

    

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation="relu"))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

    

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(256,activation="relu"))

model.add(Dense(256,activation="relu"))

model.add(Dense(10,activation="softmax"))
from keras.optimizers import RMSprop

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs =30 # Turn epochs to 30 to get 0.9967 accuracy

batch_size =10
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



datagen.fit(X_train)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose =1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)