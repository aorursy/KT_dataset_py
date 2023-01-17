# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import matplotlib.pyplot as plt

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

X_train = train.iloc[:,1:].values

y_train = train.iloc[:,0].values

X_test = train.iloc[:,1:].values

y_test = train.iloc[:,0].values
# to plot the image from training data

img = np.reshape(X_train[101] , (28,28))

plt.imshow(img)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# normalise the data

X_train = X_train/255

X_test = X_test/255
# convert labels into one hot encoded

from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)
y_train.shape
from keras import Sequential

from keras.layers import Dense,Convolution2D , MaxPooling2D, BatchNormalization,Dropout,Flatten

from keras.activations import relu
# build convolutional neural network

def cnn():

    model = Sequential([

        # 1st conv layer

        Convolution2D(32 ,kernel_size = (3,3) , input_shape = (28,28 ,1) , strides = (1,1) , padding = "SAME",activation = "relu"),

        MaxPooling2D(pool_size = (2,2) , strides = (2,2)),

        BatchNormalization(),

        Dropout(0.3),

        

        #2nd conv layer

        Convolution2D(64 ,kernel_size = (3,3) , strides = (1,1) , padding = "SAME"),

        MaxPooling2D(pool_size = (2,2) , strides = (2,2)),

        BatchNormalization(),

        Dropout(0.3),

        # 3rd conv layer

        Convolution2D(64 , kernel_size = (3,3) , strides = (1,1) , padding = "SAME"),

        MaxPooling2D(pool_size = (2,2) , strides = (2,2)),

        BatchNormalization(),

        Dropout(0.3),

        

        # Flatten the output

        Flatten(),

        

        # fully connected dense layer

        Dense(128 , activation = 'relu'),

        Dense(10 , activation = 'softmax')

        

    ])

    

    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    

    return model
classifier = cnn()

print(classifier.summary())
classifier.optimizer.lr = 0.001
history = classifier.fit(X_train , y_train , batch_size = 35 , epochs = 20 , validation_split = 0.2 , verbose = 1)
plt.plot(history.history['loss'] , 'green')

plt.plot(history.history['val_loss'] , 'red')

plt.xlabel('Epochs')
plt.plot(history.history['acc'] , 'blue')

plt.plot(history.history['val_acc'] , 'orange')