# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import MaxPooling2D, Activation, Conv2D, Dense, Dropout, Flatten

from keras import optimizers

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)  



train = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/train',

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 color_mode='grayscale',

                                                 class_mode = 'binary')



val = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/val',

                                                target_size=(64, 64),

                                                batch_size=32,

                                                color_mode='grayscale',

                                                class_mode='binary')



test = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/test',

                                                target_size = (64, 64),

                                                batch_size = 32,

                                                color_mode='grayscale',

                                                class_mode = 'binary')
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=(64,64,1),activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(100,activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',loss="binary_crossentropy", metrics=["accuracy"])
cnn_model = model.fit_generator(train, steps_per_epoch=163, epochs=10, verbose=1, callbacks=None, validation_data=val, validation_steps=624,class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
test_accu = model.evaluate_generator(test,steps=624)

print(test_accu[1]*100)
plt.plot(cnn_model.history['acc'])

plt.plot(cnn_model.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()
plt.plot(cnn_model.history['val_loss'])

plt.plot(cnn_model.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()