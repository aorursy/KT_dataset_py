# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import tensorflow as tf

import keras

from keras import Sequential

from keras.layers import GlobalAveragePooling2D,Dropout,Dense,Conv2D,BatchNormalization,MaxPooling2D,Flatten,Activation

from tqdm import tqdm_notebook

from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

from keras.preprocessing.image import ImageDataGenerator, load_img

import tensorflow.keras.applications.efficientnet as efn 
physical_devices = tf.config.list_physical_devices('GPU')

try:

    tf.config.experimental.set_memory_growth(physical_devices[0], True)

except:

  # Invalid device or cannot modify virtual devices once initialized.

    pass

!nvidia-smi
os.mkdir('./models/')

os.mkdir('./models/efficient_Net/')

os.mkdir('./models/efficient_Net/csv_logs/')
TRAIN_PATH = '/kaggle/input/nsfw-image-classification/train/'

VAL_PATH =  '/kaggle/input/nsfw-image-classification/val/'

TEST_PATH = '/kaggle/input/nsfw-image-classification/test/'
IMAGE_WIDTH=224

IMAGE_HEIGHT=224

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3

BATCH_SIZE=32

NB_EPOCHS=300
cats = ['neutral','sexy','porn','drawings','hentai']
efficient_model = efn.EfficientNetB4(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))  # or weights='noisy-student'



x = efficient_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

x = Dense(5, activation='softmax')(x)

model = keras.Model(inputs=efficient_model.input, outputs=x)



model.summary()
model.compile(loss='categorical_crossentropy', 

              optimizer='SGD', 

              metrics=['accuracy'])    
TIME_STR = time.asctime( time.localtime(time.time()) ).replace(' ', '_')

CSV_PATH = "./models/efficient_Net/csv_logs/kaggle_nsfw-" + TIME_STR

csv_logger = CSVLogger(CSV_PATH, separator=',', append= True)

    

earlystop = EarlyStopping(monitor = 'val_acc', patience=20)





filepath = "./models/efficient_Net/kaggle_nsfw-{epoch:02d}-t{accuracy:.2f}-v{val_accuracy:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.1,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.,

)



train_generator = train_datagen.flow_from_directory(

    TRAIN_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical'

)



validation_generator = train_datagen.flow_from_directory(

    VAL_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='categorical'

)
model.fit_generator(

    train_generator,

    steps_per_epoch = train_generator.samples // BATCH_SIZE,

    validation_data = validation_generator, 

    validation_steps = validation_generator.samples // BATCH_SIZE,

    epochs = NB_EPOCHS,

    callbacks=[csv_logger, earlystop, checkpoint])