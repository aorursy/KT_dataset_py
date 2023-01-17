# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





from __future__ import absolute_import

from __future__ import print_function

from keras.utils import np_utils # For y values



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





import os

print(os.listdir("../input"))

import pandas as pd

import numpy as np





# For plotting

%matplotlib inline

import seaborn as sns

# For Keras

from keras.models import Sequential

from keras.layers.core import Dense, Dropout

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt
os.listdir("../input/mnist-in-csv")
train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
train.head()
#Remove the first column from the data, as it is the label and put the rest in X

X_train = train.iloc[:, 1:].values/255.#.reshape(-1,28,28,1)

#Remove everything except the first column from the data, as it is the label and put it in y

y_train = train.iloc[:, :1].values
#Remove the first column from the data, as it is the label and put the rest in X

X_test = test.iloc[:, 1:].values/255.#.reshape(-1,28,28,1)

#Remove everything except the first column from the data, as it is the label and put it in y

y_test = test.iloc[:, :1].values
X_train.shape
#set input to the shape of one X value

dimof_input = X_train.shape[1]



# Set y categorical

dimof_output = int(np.max(y_train)+1)

Y_train = np_utils.to_categorical(y_train, dimof_output)

Y_test = np_utils.to_categorical(y_test, dimof_output)
def viz(history):

    # Plot training & validation accuracy values

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Valid'], loc='upper left')

    plt.show()



    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Valid'], loc='upper left')

    plt.show()
from keras.callbacks import EarlyStopping

ES = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
model = Sequential()

model.add(Dense(196, input_dim=dimof_input, kernel_initializer='uniform', activation='relu'))

model.add(Dense(356,kernel_initializer='uniform', activation='relu'))

model.add(Dense(196,kernel_initializer='uniform', activation='relu'))

model.add(Dense(dimof_output, kernel_initializer='uniform', activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history1 = model.fit(

    X_train, Y_train,

    validation_split=0.2,

    batch_size=500, epochs=50, verbose=1, callbacks = [ES])
model.evaluate(X_test,Y_test)
from keras import regularizers



cnn_model = Sequential()



#same is zero padding

#default strides is strides=(1, 1)

#strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. 

cnn_model.add(Conv2D(filters = 1, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))



#cnn_model.add(MaxPool2D(pool_size=(2,2)))



cnn_model.add(Flatten())

cnn_model.add(Dense(256, activation = "relu"))

cnn_model.add(Dense(10, activation = "softmax"))

cnn_model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
fit4 = cnn_model.fit(

    X_train.reshape(-1,28,28,1), Y_train,

    validation_split=0.2,

    batch_size=500, epochs=50, verbose=1, callbacks = [ES])
cnn_model.evaluate(X_test.reshape(-1,28,28,1),Y_test)
#http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

viz(fit4)
'''

from keras import regularizers



cnn_model = Sequential()



cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

cnn_model.add(MaxPool2D(pool_size=(2,2)))

#cnn_model.add(Dropout(0.25))





cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

                 

#For max pooling strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

#cnn_model.add(Dropout(0.25))





cnn_model.add(Flatten())

cnn_model.add(Dense(256, activation = "relu"))

#cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(10, activation = "softmax"))

cnn_model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

'''
'''

from keras import regularizers



cnn_model = Sequential()



cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))



#For max pooling strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size

cnn_model.add(MaxPool2D(pool_size=(2,2)))





cnn_model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))





cnn_model.add(Flatten())

cnn_model.add(Dense(256, activation = "relu"))



#Probaility that each neuron could be dropped

cnn_model.add(Dropout(0.5))

cnn_model.add(Dense(10, activation = "softmax"))

cnn_model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])



'''