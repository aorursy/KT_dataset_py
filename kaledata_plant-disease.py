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
# Importing Keras libraries and packages

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization



# Initializing the CNN

classifier = Sequential()



# Convolution Step 1

classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))



# Max Pooling Step 1

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

classifier.add(BatchNormalization())



# Convolution Step 2

classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))



# Max Pooling Step 2

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))

classifier.add(BatchNormalization())



# Convolution Step 3

classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))

classifier.add(BatchNormalization())



# Convolution Step 4

classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))

classifier.add(BatchNormalization())



# Convolution Step 5

classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))



# Max Pooling Step 3

classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))

classifier.add(BatchNormalization())



# Flattening Step

classifier.add(Flatten())



# Full Connection Step

classifier.add(Dense(units = 4096, activation = 'relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(units = 4096, activation = 'relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(units = 1000, activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(BatchNormalization())

classifier.add(Dense(units = 38, activation = 'softmax'))

classifier.summary()
classifier.load_weights('../input/plant-diseases-classification-using-alexnet/best_weights_9.hdf5')

from keras import optimizers

classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# image preprocessing

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   fill_mode='nearest')



valid_datagen = ImageDataGenerator(rescale=1./255)



batch_size = 128

base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"



training_set = train_datagen.flow_from_directory(base_dir+'/train',

                                                 target_size=(224, 224),

                                                 batch_size=batch_size,

                                                 class_mode='categorical')



valid_set = valid_datagen.flow_from_directory(base_dir+'/valid',

                                            target_size=(224, 224),

                                            batch_size=batch_size,

                                            class_mode='categorical')
train_num = training_set.samples

valid_num = valid_set.samples
# checkpoint

from keras.callbacks import ModelCheckpoint

weightpath = "best_weights_9.hdf5"

checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

callbacks_list = [checkpoint]



#fitting images to CNN

history = classifier.fit_generator(training_set,

                         steps_per_epoch=train_num//batch_size,

                         validation_data=valid_set,

                         epochs=1,

                         validation_steps=valid_num//batch_size,

                         callbacks=callbacks_list)

#saving model

filepath="AlexNetModel.hdf5"

classifier.save(filepath)