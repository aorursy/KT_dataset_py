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
train  = pd.read_csv("../input/sign-language-mnist/sign_mnist_train.csv")

test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test.csv")
test.shape
y_train_label=train.iloc[:,0]

y_test_label=test.iloc[:,0]

y_train_label.shape
y_test_label.shape
train = train.iloc[:,1:]

test = test.iloc[:,1:]
train_x = train.values

test_x = test.values

train_x
train_x = train_x - np.mean(train_x) / train_x.std()

test_x = test_x - np.mean(test_x) / train_x.std()
test_x
train_x.shape
train_x = train_x.reshape(-1,28,28,1)

test_x = test_x.reshape(-1,28,28,1)
import pandas as pd

import numpy as np

import keras

import cv2

from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential

from keras import optimizers

from keras import backend as K

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
w=10

h=10

fig=plt.figure(figsize=(8,8))

columns = 5

rows = 5

for i in range(1, columns*rows +1):

    img = train_x[i]

    fig.add_subplot(rows, columns, i)

    plt.imshow(np.squeeze(img))

plt.show()
LB = LabelBinarizer()

y_train_label = LB.fit_transform(y_train_label)

y_test_label = LB.fit_transform(y_test_label)

y_train_label.shape
y_train_label
xtrain,xval,ytrain,yval=train_test_split(train_x, y_train_label,train_size=0.75,random_state=0)
datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False,    

        rotation_range=15,    

        width_shift_range=0.1,

        height_shift_range=0.1,  

        horizontal_flip=True,  

        vertical_flip=False)
datagen.fit(xtrain)
import tensorflow as tf
model = Sequential()

 

model.add(Conv2D(32, (3, 3), padding = "same", activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

 

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(24, activation='softmax'))
learning_rate = 1e-3

lr_decay = 1e-6

sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

print(xtrain.shape, xval.shape)
history = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=32),

                    steps_per_epoch=xtrain.shape[0]//32,

                    epochs=100,

                    verbose=1,

                    validation_data=(xval, yval))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Training Accuracy vs Validation Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Training Loss vs Validation Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()