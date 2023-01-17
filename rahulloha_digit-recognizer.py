# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing important modules

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np

#Installing Tensorboard

!pip install Tensorboard
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
x = train.drop(['label'], axis =1)

y = train['label']


from sklearn.model_selection import train_test_split

(x_train,x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3,random_state = 101, shuffle=True)
#Verify the shape

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
#Reshape the data

x_train = x_train.values.reshape(x_train.shape[0],28,28,1)



x_test = x_test.values.reshape(x_test.shape[0],28,28,1)



#Change into float32 datatype and Normalize x_train and x_test by dividing it by 255.0



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



#Normalizing the input

x_train /= 255.0

x_test /= 255.0

#Verify the shape of x_train and x_test

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print('X_test shape:', x_test.shape)

print(x_test.shape[0], 'test samples')
#Using One-hot encoding to divide y_train and y_test into required no of output classes



num_classes = 10

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
#Verify shape of y_train

print('y_train shape:', y_train.shape)
#Initialize the model

model = Sequential()



#Add a Convolutional Layer with 32 filters of size 3X3 and activation function as 'ReLU' 

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu'))



#Add a Convolutional Layer with 64 filters of size 3X3 and activation function as 'ReLU' 

model.add(Conv2D(64, (3, 3), activation='relu'))



#Add a MaxPooling Layer of size 2X2 

model.add(MaxPooling2D(pool_size=(2, 2)))



#Apply Dropout with 0.25 probability 

model.add(Dropout(0.25))



#Flatten the layer

model.add(Flatten())



#Add Fully Connected Layer with 128 units and activation function as 'ReLU'

model.add(Dense(128, activation='relu'))



#Apply Dropout with 0.5 probability 

model.add(Dropout(0.5))



#Add Fully Connected Layer with 10 units and activation function as 'softmax'

model.add(Dense(num_classes, activation='softmax'))
from keras.optimizers import Adam

from keras.losses import categorical_crossentropy



#To use adam optimizer for learning weights with learning rate = 0.001

optimizer = Adam(lr=0.000003)

#Set the loss function and optimizer for the model training

model.compile(loss=categorical_crossentropy,

              optimizer=optimizer,

              metrics=['accuracy'])
model.fit(x_train, y_train,

          batch_size=64,

          epochs=120,validation_data=(x_test, y_test))#Testing the model on test set

score = model.evaluate(x_test, y_test)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
#Loading test File

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test = test.values.reshape(test.shape[0],28,28,1)

test = test.astype('float32')

test /= 255.0
predictions = model.predict_classes(test, verbose=1)

df = pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),

              "Label":predictions})

df
df.to_csv('sample_submission_1.csv', index=False,header=True)
for layers in model.layers:

    print(layers.name)

    if('dense' not in layers.name):

        layers.trainable = False

        print(layers.name + 'is not trainable\n')

    if('dense' in layers.name):

        print(layers.name + ' is trainable\n')