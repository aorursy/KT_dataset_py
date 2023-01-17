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
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Activation, ZeroPadding2D, Flatten

from keras.regularizers import l2

from keras.losses import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import numpy as np

import csv
def conv_layer(model, num_kernels, kernel_size=(3,3), strides=1, activation='relu', pad_mode='same',

               addBatchNorm=True, addPooling=True):

    """

    Builds the layers of convolutional neural networks with default (3,3) max pooling.



    :param model: The object of keras.models.Sequential

    :param num_kernels: Number of kernels

    :param kernel_size: Size of kernels

    :param strides: Strides of conv. layer

    :param activation: Name of activation function

    :param pad_mode: 'same'/'valid'

    :param addBatchNorm: True/False

    :param addPooling: True/False

    :return: return an object of model.

    """

    model.add(Conv2D(num_kernels, kernel_size=kernel_size, strides=strides, padding=pad_mode,

                     kernel_regularizer=l2()))

    if addBatchNorm:

        model.add(BatchNormalization(axis=3))



    model.add(Activation(activation))



    if addPooling:

        model.add(MaxPooling2D((3,3), strides=strides))



    return model
num_classes = 10

batch_size = 256

epochs = 200

img_x, img_y = (28, 28)

input_shape = (img_x, img_y, 1)
train_dataset1 = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

train_dataset2 = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

train_dataset = train_dataset1.append(train_dataset2, ignore_index = True)

test_dataset = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')



train_dataset.groupby(by='label').size()
x_MNIST = train_dataset.drop(['label'], axis=1).values.reshape(-1, img_x, img_y, 1).astype('float32')/255

y_MNIST = train_dataset['label']

x_test = test_dataset.drop(["id"], axis=1).values.reshape(-1, img_x, img_y, 1).astype('float32')/255
x_train, x_dev, y_train_origin, y_dev_origin = train_test_split(x_MNIST, y_MNIST, test_size=0.1, random_state=0)
y_train = np_utils.to_categorical(y_train_origin, num_classes=num_classes)

y_dev = np_utils.to_categorical(y_dev_origin, num_classes=num_classes)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=5, factor=np.sqrt(0.5), verbose=2)
img_gen = ImageDataGenerator( featurewise_center=False,

                              samplewise_center=False,

                              featurewise_std_normalization=False,

                              samplewise_std_normalization=False,

                              zca_whitening=False,

                              rotation_range=10,

                              zoom_range=0.10,

                              width_shift_range=0.1,

                              height_shift_range=0.1,

                              horizontal_flip=False,

                              vertical_flip=False)
model = Sequential()

model.add(ZeroPadding2D((1,1), input_shape=input_shape))

# Group 1

model = conv_layer(model, 32, kernel_size=(5, 5), strides=1, activation='relu', pad_mode='valid', addPooling=True)

model = conv_layer(model, 32, kernel_size=(5, 5), strides=1, activation='relu', pad_mode='same', addPooling=True)

model = conv_layer(model, 32, kernel_size=(5, 5), strides=1, activation='relu', pad_mode='valid', addPooling=True)

model.add(Dropout(0.2))

# Group 2

model = conv_layer(model, 64, kernel_size=(3, 3), strides=1, activation='relu', pad_mode='valid', addPooling=True)

model = conv_layer(model, 128, kernel_size=(3, 3), strides=1, activation='relu', pad_mode='same', addPooling=False)

model = conv_layer(model, 128, kernel_size=(3, 3), strides=1, activation='relu', pad_mode='same', addPooling=False)

model = conv_layer(model, 128, kernel_size=(3, 3), strides=1, activation='relu', pad_mode='valid', addPooling=False)

model.add(Dropout(0.2))

# Group 3

model = conv_layer(model, 256, kernel_size=(3, 3), strides=1, activation='relu', pad_mode='valid', addPooling=True)



model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(optimizer=Adam(),

              loss=categorical_crossentropy,

              metrics=['accuracy'])
print(model.summary())

print('Size of training dataset:', x_train.shape[0])

print('Size of dev. dataset:    ', x_dev.shape[0])
img_gen.fit(x_train)

model.fit_generator(img_gen.flow(x_train, y_train, batch_size=batch_size),

                    epochs=epochs,

                    steps_per_epoch=x_train.shape[0] // batch_size,

                    verbose=2,

                    validation_data=(x_dev, y_dev),

                    callbacks=[lr_reduction])
print('\nPredicting...')

predictions = model.predict(x_test)

print('Prediction completed.')

predictions = np.argmax(predictions, axis=1)
with open('07.Kaggle_submission.csv', 'w', newline='') as csv_file:

    print('\nSaving file...')

    csv_writer = csv.writer(csv_file, delimiter=',')

    # Define column name.

    csv_writer.writerow(['id', 'label'])

    for i in range(len(predictions)):

        csv_writer.writerow([i, predictions[i]])



    print('File: 07.Kaggle_submission.csv Saved completed.')
from sklearn.metrics import confusion_matrix

import seaborn as sn

import matplotlib.pyplot as plt



def plot_confusion_matrix(cm, num_classes, title='Confusion matrix'):

    print('Title: ', title)

    print(cm)

    

## Show confusion matrix

# Training dataset

predict_train = model.predict(x_train)

# Convert the one-hot vectors to the corresponding classes.

predict_train = np.argmax(predict_train, axis=1)

y_train = np.argmax(y_train, axis=1)

# Build confusion matrix.

confusion_mx = confusion_matrix(y_train, predict_train)

plot_confusion_matrix(confusion_mx, num_classes=num_classes, title='Confusion matrix - training set')



# Developing dataset

predict_dev = model.predict(x_dev)

# Convert the one-hot vectors to the corresponding classes.

predict_dev = np.argmax(predict_dev, axis=1)

y_dev = np.argmax(y_dev, axis=1)

# Build confusion matrix.

confusion_mx = confusion_matrix(y_dev, predict_dev)

plot_confusion_matrix(confusion_mx, num_classes=num_classes, title='Confusion matrix - developing set')
