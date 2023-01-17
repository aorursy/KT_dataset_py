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
import numpy as np

np.random.seed(2017)  # for reproducibility



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras import backend as K
batch_size = 128

nb_classes = 5

nb_epoch = 5



# input image dimensions

img_rows, img_cols = 28, 28



# number of convolutional filters to use

nb_filters = 32



# size of pooling area for max pooling

pool_size = 2



# convolution kernel size

kernel_size = 3



input_shape = (img_rows, img_cols, 1)
# read training data from CSV file 

train = pd.read_csv('../input/train.csv')

train.head()
X = train.iloc[:,1:] # Features

y = train['label']   # Target



# split data into train and test

from sklearn.cross_validation import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# create two datasets one with digits below 5 and one with 5 and above

X_train_lt5 = X_train[y_train < 5]

y_train_lt5 = y_train[y_train < 5]

X_test_lt5 = X_test[y_test < 5]

y_test_lt5 = y_test[y_test < 5]



X_train_gte5 = X_train[y_train >= 5]

y_train_gte5 = y_train[y_train >= 5] - 5       # make classes start at 0 for

X_test_gte5 = X_test[y_test >= 5]              # np_utils.to_categorical

y_test_gte5 = y_test[y_test >= 5] - 5
def train_model(model, train, test, nb_classes):

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

    X_train = train[0].values.reshape(-1, 28,28,1)

    X_test = test[0].values.reshape(-1, 28,28,1)

    X_train = X_train.astype('float32')

    X_test = X_test.astype('float32')

    X_train /= 255

    X_test /= 255

    print('X_train shape:', X_train.shape)

    print(X_train.shape[0], 'train samples')

    print(X_test.shape[0], 'test samples')



    # convert class vectors to binary class matrices

    Y_train = np_utils.to_categorical(train[1], nb_classes)

    Y_test = np_utils.to_categorical(test[1], nb_classes)



    model.compile(loss='categorical_crossentropy',

                  optimizer='adadelta',

                  metrics=['accuracy'])



    model.fit(X_train, Y_train,

              batch_size=batch_size, nb_epoch=nb_epoch,

              verbose=1,

              validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)

    print('Test score:', score[0])

    print('Test accuracy:', score[1])
# define two groups of layers: feature (convolutions) and classification (dense)

feature_layers = [

    Convolution2D(nb_filters, kernel_size, kernel_size,

                  border_mode='valid',

                  input_shape=input_shape),

    Activation('relu'),

    Convolution2D(nb_filters, kernel_size, kernel_size),

    Activation('relu'),

    MaxPooling2D(pool_size=(pool_size, pool_size)),

    Dropout(0.25),

    Flatten(),

]

classification_layers = [

    Dense(128),

    Activation('relu'),

    Dropout(0.5),

    Dense(nb_classes),

    Activation('softmax')

]



# create complete model

model = Sequential(feature_layers + classification_layers)



# train model for 5-digit classification [0..4]

train_model(model, (X_train_lt5, y_train_lt5), (X_test_lt5, y_test_lt5), nb_classes)
# freeze feature layers and rebuild model

for layer in feature_layers:

    layer.trainable = False



# transfer: train dense layers for new classification task [5..9]

train_model(model, (X_train_gte5, y_train_gte5), (X_test_gte5, y_test_gte5), nb_classes)