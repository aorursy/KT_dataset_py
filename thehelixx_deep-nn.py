# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import keras

# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline







from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.optimizers import Adam

from keras.optimizers import Adamax



from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from matplotlib.pyplot import plot

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.image as mpimg



test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
train.head()
test.head()
y_train = train["label"]

y_train.head()
x_train = train.drop(labels = ["label"],axis = 1) 
y_train.value_counts()
x_train.isnull().describe()
x_train = x_train/255.0

test = test/255.0
x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print('x_train shape:', x_train.shape)
y_train = keras.utils.to_categorical(y_train,num_classes=10)
random_seed = 1
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)
a= plt.imshow(x_train[1][:,:,0])
from keras.layers import LeakyReLU
model = Sequential()



model.add(Conv2D(32,(3,3), padding='Same', activation='relu', input_shape=(28, 28, 1)))

model.add(Dropout(0.25))

model.add(Conv2D(32,(7,7),activation='relu'))

model.add(Conv2D(128,(5,5),activation='relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3), padding='Same',activation='relu'))

model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))











model.add(Flatten())

model.add(Dense(128, activation='linear'))

model.add(LeakyReLU(alpha=.001))

model.add(Dense(64, activation='linear'))

model.add(LeakyReLU(alpha=.001))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(324, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(32, activation='linear'))

model.add(LeakyReLU(alpha=.001))



model.add(Dense(10, activation='softmax'))

model.summary()
optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(

        featurewise_center=False,  # Set input mean to 0 over the dataset

        samplewise_center=False,  # Set each sample mean to 0

        featurewise_std_normalization=False,  # Divide inputs by std of the dataset

        samplewise_std_normalization=False,  # Divide each input by its std

        zca_whitening=False,  # Apply ZCA whitening

        rotation_range=20,  # Randomly rotate images in the range (degrees, 0 to 180) # 10

        zoom_range = 0.13, # Randomly zoom image # 0.1

        width_shift_range=0.1,  # Randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # Randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # Randomly flip images

        vertical_flip=False)  # Randomly flip images



datagen.fit(x_train)
model.fit_generator(

    datagen.flow(x_train, y_train, batch_size=256),

    steps_per_epoch=len(x_train)//256,

    epochs=30,

    

)
#history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)
score = model.evaluate(x_val, y_val, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
result = model.predict(test)

result = np.argmax(result,axis=1)

result = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)



submission.to_csv("cnn_mnist.csv",index=False)