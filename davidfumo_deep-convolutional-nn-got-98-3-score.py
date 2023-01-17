from __future__ import print_function

import numpy as np

import pandas as pd

np.random.seed(1337)  # for reproducibility

%matplotlib inline

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras import backend as K
# getting the datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)
# plot some digits

for i in range(9):

    plt.subplot(3,3,i+1)

    data =train.ix[i, 'pixel0':'pixel783'].reshape(28,28)

    plt.imshow(data, cmap='gray')

    plt.axis('off')
# separate predictors and labels, transform to ndarrays

y = train[[0]].values.ravel() # labels/target

X = train.iloc[:,1:].values   # Predictors(pixels that form the image)

test = test.values
batch_size = 128

nb_classes = 10

nb_epoch = 12



# input image dimensions: 28x28 = 784 pixels (exactly what we have in training dataset)

img_rows, img_cols = 28, 28

# number of convolutional filters to use

nb_filters = 32

# size of pooling area for max pooling

pool_size = (2, 2)

# convolution kernel size

kernel_size = (3, 3)
# Split our data into train and test, this way will be able to evaluate the perfomance of the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



if K.set_image_dim_ordering == 'th':

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    test = test.reshape(test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    test = test.reshape(test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)



# convert to array, specify data type, and reshape

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

test = test.astype('float32')



# Normalize data to have zero mean and unit variance. 

X_train /= 255

X_test /= 255

test /= 255



print('X_train shape:', X_train.shape)

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = np_utils.to_categorical(y_train, nb_classes)

y_test = np_utils.to_categorical(y_test, nb_classes)

print("One hot encoding: {}".format(y_train[0, :]))
# The Sequential model is a linear stack of layers

model = Sequential()



# in the first layer, you must specify the expected input data shape

# input: 28x28 images with 1 channel -> (1, 28, 28) tensors.

# this applies 32 (nb_filters) convolution filters of size 3x3 each.

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],

                        border_mode='valid',

                        input_shape=input_shape))

model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))



model.add(Flatten())



# Dense(128) is a fully-connected layer with 128 hidden units.

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))





# a loss function. This is the objective that the model will try to minimize.

# It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse),

# or it can be an objective function

model.compile(loss='categorical_crossentropy',

              optimizer='adadelta',

              metrics=['accuracy'])
# train the model, iterating on the data in batches

# of 128 samples(batch_size declared above)

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,

          verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])
# Make new predictions with the model on the kaggle test dataset

predictions = model.predict_classes(test, batch_size=batch_size)
submission = pd.DataFrame({

        "ImageId": np.arange(1,len(test)+1),

        "Label": predictions

    })

submission.to_csv('submission_cnn.csv', index=False)