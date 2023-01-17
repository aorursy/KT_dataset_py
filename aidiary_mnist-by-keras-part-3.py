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
from keras.utils.np_utils import to_categorical
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



X_train = train.iloc[:, 1:].values.astype('float32')

y_train = train.iloc[:, 0].values.astype('int32')

X_test = test.values.astype('float32')



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)



print(X_train.shape, X_test.shape)
# Normalize

X_train = X_train / 255.0

X_test = X_test / 255.0
y_train = to_categorical(y_train)

y_train.shape
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rotation_range=8,

    width_shift_range=0.08,

    shear_range=0.3,

    height_shift_range=0.08,

    zoom_range=0.08

)
from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',

                                            patience=3,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.00001)
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, BatchNormalization
model = Sequential()



model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
epochs = 30

batch_size = 64



history = model.fit_generator(generator=datagen.flow(X_train, y_train),

                              steps_per_epoch=X_train.shape[0] // batch_size,

                              epochs=epochs,

                              validation_data=datagen.flow(X_val, y_val),

                              validation_steps=X_val.shape[0] // batch_size)
predictions = model.predict_classes(X_test, verbose=0)

submissions = pd.DataFrame({'ImageID': list(range(1, len(predictions) + 1)),

                            'Label': predictions})

submissions.to_csv('cnn_part3.csv', index=False, header=True)