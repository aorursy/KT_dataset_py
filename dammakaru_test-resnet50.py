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
TRAIN_DIR = '/kaggle/input/datasetv2/dataset/'

TEST_DIR = '/kaggle/input/datasetv2/dataset/'
from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.applications.resnet50 import ResNet50, preprocess_input



CLASSES = 2

    

# setup model

base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output

x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.4)(x)

predictions = Dense(CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

   

# transfer learning

for layer in base_model.layers:

    layer.trainable = False
model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator



WIDTH = 224

HEIGHT = 224

BATCH_SIZE = 8



# data prep

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    class_mode='categorical')

    

validation_generator = validation_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    class_mode='categorical')
EPOCHS = 20

BATCH_SIZE = 8

STEPS_PER_EPOCH = 99

VALIDATION_STEPS = 64



MODEL_FILE = 'image_classifier_vgg16.model'



history = model.fit_generator(

    train_generator,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=validation_generator,

    validation_steps=VALIDATION_STEPS)
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Resnet50 model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Resnet50 model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()