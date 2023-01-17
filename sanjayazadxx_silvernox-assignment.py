# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



path_train = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train"

path_val = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val"

path_test = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test"



# Any results you write to the current directory are saved as output.
import keras

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)

val_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    path_train,

    target_size=(150, 150),

    batch_size=16,

    class_mode='categorical')



test_generator = test_datagen.flow_from_directory(

    path_test,

    target_size=(150, 150),

    batch_size=16,

    class_mode='categorical')



validation_generator = val_datagen.flow_from_directory(

    path_val,

    target_size=(150, 150),

    batch_size=16,

    class_mode='categorical')
from keras import layers, models

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from keras.callbacks import EarlyStopping, ModelCheckpoint



'''Model 1'''

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Dropout(0.3))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.3))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Dropout(0.3))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dense(2, activation='sigmoid'))

model.summary()
model.compile(

    loss='binary_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)



result = model.fit_generator(

    train_generator, 

    epochs=100,

    steps_per_epoch=train_generator.samples / train_generator.batch_size,

    validation_data=validation_generator,

    validation_steps=validation_generator.samples / validation_generator.batch_size,

    callbacks=[

        EarlyStopping(

            monitor='val_loss', 

            mode='auto', 

            verbose=1, 

            patience=5,

        ), 

        ModelCheckpoint(

            'best_model.h5', 

            monitor='val_loss', 

            mode='auto', 

            save_best_only=True, 

            verbose=1

        )

    ]

)
test_loss, test_score = model.evaluate_generator(test_generator,steps=100)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
# from keras.optimizers import SGD

# model = Sequential()



# '''Model 2 - VGG Like convnet'''

# # input: 100x100 images with 3 channels -> (150, 150, 3) tensors.

# # this applies 32 convolution filters of size 3x3 each.

# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

# model.add(Conv2D(32, (3, 3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Conv2D(64, (3, 3), activation='relu'))

# model.add(Conv2D(64, (3, 3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



# model.add(Flatten())

# model.add(Dense(256, activation='relu'))

# model.add(Dropout(0.5))

# model.add(Dense(2, activation='softmax'))



# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy', optimizer=sgd)
# model.compile(

#     loss='binary_crossentropy',

#     optimizer='adam',

#     metrics=['accuracy']

# )



# result = model.fit_generator(

#     train_generator, 

#     epochs=100,

#     steps_per_epoch=train_generator.samples / train_generator.batch_size,

#     validation_data=validation_generator,

#     validation_steps=validation_generator.samples / validation_generator.batch_size,

#     callbacks=[

#         EarlyStopping(

#             monitor='val_loss', 

#             mode='auto', 

#             verbose=1, 

#             patience=5,

#         ), 

#         ModelCheckpoint(

#             'best_model.h5', 

#             monitor='val_loss', 

#             mode='auto', 

#             save_best_only=True, 

#             verbose=1

#         )

#     ]

# )
# test_loss, test_score = model.evaluate_generator(test_generator,steps=100)

# print("Loss on test set: ", test_loss)

# print("Accuracy on test set: ", test_score)