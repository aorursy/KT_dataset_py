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
import keras

from matplotlib import pyplot as plt

%matplotlib inline



import numpy as np

np.random.seed(2017)



from keras import backend as K

from keras.models import Sequential

from keras.datasets import mnist

from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten

from keras.utils import np_utils

from keras.preprocessing import sequence

keras.backend.backend()

keras.backend.image_dim_ordering()



# Ensure to set the image dimension appropriately

# To start Kears with Theano backend please run the following command while starting jupter notebook.

# KERAS_BACKEND=theano jupyter notebook

K = keras.backend.backend()

if K=='tensorflow':

    keras.backend.set_image_dim_ordering('tf')

else:

    keras.backend.set_image_dim_ordering('th')
img_rows, img_cols = 28, 28

nb_classes = 10



nb_filters = 5 # the number of filters

nb_pool = 2 # window size of pooling

nb_conv = 3 # window or kernel size of filter

nb_epoch = 10



input_shape = (img_rows, img_cols, 1)



# read training data from CSV file 

train = pd.read_csv('../input/train.csv')

train.head()



X = train.iloc[:,1:] # Features

y = train['label']   # Target



# split data into train and test

from sklearn.cross_validation import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



X_train = X_train.values.reshape(-1, img_rows, img_cols,1) 

X_test = X_test.values.reshape(-1, img_rows, img_cols,1)

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255

print('X_train shape:', X_train.shape)

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)
# define two groups of layers: feature (convolutions) and classification (dense)

feature_layers = [

    Convolution2D(nb_filters, nb_conv, nb_conv, input_shape=input_shape),

    Activation('relu'),

    Convolution2D(nb_filters, nb_conv, nb_conv),

    Activation('relu'),

    MaxPooling2D(pool_size=(nb_pool, nb_pool)),

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



model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])

print(model.summary())
model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=256, verbose=2,  validation_split=0.2)
# read test data from CSV file 

test = pd.read_csv('../input/test.csv').values



X_test = test.reshape(-1, 28,28,1)

X_test = X_test.astype('float32')

X_test /= 255

print(X_test.shape, 'test samples')



# predict results

predictions = model.predict_classes(X_test)

print(predictions)



# Create submission file

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("cnn_mnist.csv", index=False, header=True)

submissions.head()