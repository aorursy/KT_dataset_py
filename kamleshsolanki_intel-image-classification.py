# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras import Sequential

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
train      = '/kaggle/input/intel-image-classification/seg_train/seg_train'

validation = '/kaggle/input/intel-image-classification/seg_test/seg_test'

test       = '/kaggle/input/intel-image-classification/seg_pred'
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   featurewise_center=True,

                                    featurewise_std_normalization=True,

                                    rotation_range=20,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale = 1./255,

                                   featurewise_center=True,

                                    featurewise_std_normalization=True,

                                    rotation_range=20,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    horizontal_flip=True)



train_generator = train_datagen.flow_from_directory(train,

                                                   target_size = (150, 150),

                                                   batch_size = 32,

                                                   class_mode='categorical')



validation_generator = validation_datagen.flow_from_directory(validation,

                                                   target_size = (150, 150),

                                                   batch_size = 32,

                                                   class_mode='categorical')





test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(test, target_size = (150, 150), batch_size = 1)



class_names = train_generator.class_indices

class_names = {i : class_name for i, class_name in enumerate(class_names)}

print(class_names)
for img_batch,label_batch in train_generator:

    plt.figure(figsize=(20,10))

    for ix in range(32):

        sub = plt.subplot(4, 8, ix + 1)

        plt.imshow(img_batch[ix])

        plt.xticks([])

        plt.yticks([])

        plt.xlabel(class_names[np.argmax(label_batch[ix])])

    break

        
def build_model(input_shape = (150, 150, 3)):

    model = Sequential()

    model.add(Conv2D(64, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu', input_shape = input_shape))

    model.add(Conv2D(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))

    model.add(MaxPool2D())

    model.add(Conv2D(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))

    model.add(Conv2D(512, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu'))  

    model.add(MaxPool2D())

    model.add(Flatten())

    model.add(Dropout(0.2))

    model.add(Dense(1048, activation = 'relu'))

    model.add(Dropout(0.2))

    model.add(Dense(len(class_names), activation = 'softmax'))

    

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    

    return model
def build_inception(input_shape = (150, 150, 3)):

    base_model = InceptionV3(input_shape = input_shape, include_top = False, weights = 'imagenet')

    base_model.trainable = True

    layers = base_model.layers

    for layer in layers[:-3]:

        layer.trainable = False

    model = Sequential()

    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(2048, activation = 'relu'))    

    model.add(Dense(1024, activation = 'relu'))

    model.add(Dropout(0.2))

    model.add(Dense(len(class_names), activation = 'softmax'))

    

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model    
model = build_inception()

model.summary()
hist = model.fit_generator(train_generator, epochs = 40, validation_data=validation_generator)
plt.figure(figsize = (15, 7))

for key in hist.history.keys():

    plt.plot(hist.history[key], label = key)

plt.legend()
model.save_weights('model_inception_weights.h5')
model = build_inception()

model.load_weights('model_inception_weights.h5')
for img_batch,label_batch in train_generator:

    plt.figure(figsize=(20,10))

    for ix in range(32):

        sub = plt.subplot(4, 8, ix + 1)

        plt.imshow(img_batch[ix])

        plt.xticks([])

        plt.yticks([])

        pred = model.predict(img_batch[ix].reshape(1, 150, 150, 3))

        class_key = np.argmax(pred)

        prob = np.max(pred) * 100

        ylabel = '{} ({:.2f}%)'.format(class_names[class_key], prob)

        plt.ylabel(ylabel)

        plt.xlabel(class_names[np.argmax(label_batch[ix])])

    break
index = 1

plt.figure(figsize=(20,10))

row = 4

col = 8

for img_batch,label_batch in test_generator:

    sub = plt.subplot(row, col, index)

    plt.imshow(img_batch[0])

    plt.xticks([])

    plt.yticks([])

    pred = model.predict(img_batch)

    class_key = np.argmax(pred)

    prob = np.max(pred) * 100

    plt.ylabel('{:.2f}%'.format(prob))

    plt.xlabel(class_names[class_key])

    index = index + 1

    if index > row * col:

        break