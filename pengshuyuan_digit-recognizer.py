# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train_label = train["label"]

train_data = train.drop(labels="label", axis=1)

del train



# normalization

train_data = train_data / 255.0

test_data = test / 255.0



# reshape data

train_data = train_data.values.reshape(-1, 28, 28, 1)

test_data = test_data.values.reshape(-1, 28, 28, 1)



# encoding lebels to one hot vectors

train_label = to_categorical(train_label, num_classes=10)



# split validation_data and validation_label from train_data and train_label

train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1, random_state=2) 
g = plt.imshow(train_data[2][:,:,0])
# build CNN model

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])



# set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                            patience=3,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.00001)

# set batch_size and epochs

batch_size = 128

epochs = 30
# data augmentation

datagen = ImageDataGenerator(featurewise_center=False,

                             samplewise_center=False, 

                             featurewise_std_normalization=False, 

                             samplewise_std_normalization=False,

                             zca_whitening=False,

                             rotation_range=10,

                             zoom_range=0.1,

                             width_shift_range=0.1,

                             height_shift_range=0.1,

                             horizontal_flip=False,

                             vertical_flip=False)

datagen.fit(train_data)
# fit the model

history = model.fit_generator(datagen.flow(train_data, train_label, batch_size=batch_size),

                              epochs=epochs, 

                              validation_data=(val_data,val_label),

                              verbose=2, 

                              steps_per_epoch=train_data.shape[0]//batch_size, 

                              callbacks=[learning_rate_reduction])

# plot the loss and accuracy curves for training and validation

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Traning loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# predict results

results = model.predict(test_data)

# select the index with the maximum probability

results = np.argmax(results, axis=1)

results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name="ImageId"),results],axis=1)

submission.to_csv('submission.csv',index=False)