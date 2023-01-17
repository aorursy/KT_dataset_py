# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras
import pandas as pd
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
# input image dimensions
img_rows, img_cols = 28, 28
nb_classes =10
train = pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
print (K.image_dim_ordering())
X_train = (train.ix[:,1:].values).astype('float32') # all pixel values
y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28)
print(X_train.shape)
#expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
print(X_test.shape)
#y_train = y_train.reshape(y_train.shape[0],1)
print(y_train.shape)
Y_train = np_utils.to_categorical(y_train, nb_classes)
print (Y_train.shape)
def create_modelCNN(nb_classes =10, nb_filters = 32, pool_size=(2,2), kernel_size = (3,3)):
    model1 = Sequential()
    model1.add(Convolution2D(nb_filters, kernel_size,
                             border_mode='valid',
                             input_shape=(28, 28, 1)))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=pool_size))
    model1.add(Convolution2D(nb_filters, kernel_size))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=pool_size))
    model1.add(Dropout(0.5))

    model1.add(Flatten())
    model1.add(Dense(100))
    model1.add(Activation('relu'))
    model1.add(Dropout(0.3))
    model1.add(Dense(nb_classes))
    model1.add(Activation('softmax'))

    model1.compile(loss='categorical_crossentropy',
                   optimizer='adadelta',
                   metrics=['accuracy'])
    return(model1)
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)

cnn1 = create_modelCNN()    
print (cnn1.summary())
batch_size = 128
nb_epoch= 10
cnn1.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)

x=X_train[123,:,:,0]
plt.imshow(x)
v=cnn1.predict(X_train)
print(v[123])
print(Y_train[123])
predictions = cnn1.predict_classes(X_test, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)