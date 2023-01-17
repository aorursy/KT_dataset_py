# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255,shear_range=0.2,
        zoom_range=0.2,width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/kaggle/input/intel-image-classification/seg_train/seg_train',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/kaggle/input/intel-image-classification/seg_test/seg_test',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')
NUM_CLASSES = 6
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 64
def model_maker():
    base_model = VGG16(include_top = False, weights = 'imagenet',input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    input = Input(shape = (IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = Conv2D(64,(3,3), activation = 'relu')(custom_model)
    custom_model = MaxPooling2D((2,2))(custom_model)
    custom_model = Flatten()(custom_model)
    custom_model = Dense(128, activation = 'relu')(custom_model)
    custom_model = Dense(256, activation = 'relu')(custom_model)
    predictions = Dense(NUM_CLASSES, activation = 'softmax')(custom_model)
    
    return Model(inputs = input, outputs = predictions)
model = model_maker()
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit_generator(train_generator, validation_steps = 5, steps_per_epoch = 8,epochs=8, verbose=1, validation_data = validation_generator)

