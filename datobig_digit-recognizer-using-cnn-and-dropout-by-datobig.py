# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras import backend as K
training_data = pd.read_csv('../input/train.csv')
testing_data = pd.read_csv('../input/test.csv')
training_data.shape
testing_data.shape
training_data[:1]
testing_data[:1]
y_train = training_data["label"].astype('int32')
y_train.head
y_train = np_utils.to_categorical(y_train)
y_train.shape
y_train
X_train = training_data.ix[:,1:].values.astype('float32')
X_train.shape
X_test = testing_data.ix[:,0:].values.astype('float32')
X_test.shape
img_rows, img_cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_train.shape
from keras import backend as K
K.image_dim_ordering()
X_test_plot = X_test.reshape(X_test.shape[0],img_rows,img_cols).astype('float32')
X_test_plot.shape
import matplotlib.pyplot as plt

# plot 4 images as gray scale

plt.subplot(221)

plt.imshow(X_test_plot[10], cmap=plt.get_cmap('gray'))

plt.subplot(222)

plt.imshow(X_test_plot[21], cmap=plt.get_cmap('gray'))

plt.subplot(223)

plt.imshow(X_test_plot[32], cmap=plt.get_cmap('gray'))

plt.subplot(224)

plt.imshow(X_test_plot[43], cmap=plt.get_cmap('gray'))

# show the plot

plt.show()
X_test.shape
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255
def large_model():

    model = Sequential()

    model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(15, 3, 3, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

    return model
K.set_image_dim_ordering('th')
num_classes = y_train.shape[1]
model = large_model()
model.fit(X_train, y_train, validation_split=0.1, nb_epoch=2, batch_size=200,verbose=2 )