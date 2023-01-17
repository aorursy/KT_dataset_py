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
from __future__ import absolute_import, division, print_function, unicode_literals

import functools

import tensorflow as tf

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout,BatchNormalization

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models

from tensorflow import keras

print(tf.__version__)
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

validation = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

train = train.append(validation)



test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

sample_submission = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
(train_label, train_features) = (train["label"], train.drop(["label"],axis=1))

(test_id, test_features) = (test["id"], test.drop(["id"],axis=1))



(train_x, train_y) = (train_features.values, train_label.values)

test_x = test_features.values
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.05, random_state=42)
reshaped_train_x = tf.reshape(train_x, [-1,28,28,1])

reshaped_valid_x = tf.reshape(valid_x, [-1,28,28,1])

reshaped_test_x = tf.reshape(test_x, [-1,28,28,1])
normalized_train_x = reshaped_train_x/255

normalized_valid_x = reshaped_valid_x/255

normalized_test_x = reshaped_test_x/255

train_labels = keras.utils.to_categorical(train_y, 10)

valid_labels = keras.utils.to_categorical(valid_y, 10)
datagen_train = ImageDataGenerator(rotation_range=15,

                             width_shift_range = 0.2,

                             height_shift_range = 0.2,

                             shear_range = 25,

                             zoom_range = 0.4,)

datagen_valid = ImageDataGenerator()
model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(256, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Dense(128))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.summary()
lr = 2e-3

batch_size=128

epochs=50



learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                            patience=5,

                                            verbose=1,

                                            factor=0.2)



earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen_train.flow(normalized_train_x, train_labels, batch_size=batch_size),

                    epochs = epochs,

                    steps_per_epoch=normalized_train_x.shape[0] // batch_size,

                    validation_data = (normalized_valid_x,valid_labels),

                    shuffle=True,

                    callbacks=[learning_rate_reduction, earlystop])
prediction = model.predict(normalized_test_x)
test['label'] = np.argmax(prediction, axis=1)



submission = test[['id', 'label']]

submission.to_csv("submission.csv",index=False)