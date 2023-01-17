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
train= pd.read_csv("../input/train.csv")

train.head(5)
test= pd.read_csv("../input/test.csv")

test.head(5)
print(train.shape, test.shape)
X_train= train.drop(['label'], axis=1)

X_train.head(5)
y_train= train['label']
X_train.shape[0]
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape

from keras.optimizers import SGD, RMSprop

from keras.utils import np_utils

from keras.regularizers import l2

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras import backend as K

x_train = train.drop(['label'], axis=1).values.astype('float32')

Y_train = train['label'].values.astype('float32')

x_test = test.values.astype('float32')



img_width, img_height = 28, 28



n_train = x_train.shape[0]

n_test = x_test.shape[0]



n_classes = 10 



x_train = x_train.reshape(n_train,1,img_width,img_height)

x_test = x_test.reshape(n_test,1,img_width,img_height)



x_train = x_train/255 #normalize from [0,255] to [0,1]

x_test = x_test/255 



y_train = np_utils.to_categorical(Y_train)
%matplotlib inline

import matplotlib.pyplot as plt



imgplot = plt.imshow(x_train[100,0,:,:,],cmap='gray')
from keras.models import Sequential

from keras.layers.convolutional import *

from keras.layers.core import Dropout, Dense, Flatten, Activation



n_filters = 64

filter_size1 = 3

filter_size2 = 2

pool_size1 = 3

pool_size2 = 1

n_dense = 128



K.set_image_dim_ordering('th')



model = Sequential()



model.add(Convolution2D(n_filters, filter_size1, filter_size1, batch_input_shape=(None, 1, img_width, img_height), activation='relu', border_mode='valid'))



model.add(MaxPooling2D(pool_size=(pool_size1, pool_size1)))



model.add(Convolution2D(n_filters, filter_size2, filter_size2, activation='relu', border_mode='valid'))



model.add(MaxPooling2D(pool_size=(pool_size2, pool_size2)))



model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(n_dense))



model.add(Activation('relu'))



model.add(Dropout(0.5))



model.add(Dense(n_classes))



model.add(Activation('softmax'))



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,

          y_train,

          batch_size=128,

          nb_epoch=2,verbose=1,

          validation_split=.2

          )
yPred = model.predict_classes(x_test,batch_size=32,verbose=1)
pd.DataFrame({"ImageId": list(range(1,len(x_test)+1)), 

              "Label": yPred}).to_csv('MNIST-submission_1.csv', index=False,header=True)