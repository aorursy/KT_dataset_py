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
from glob import glob
import os
import numpy as np
import pandas as pd
import random
from skimage.io import imread
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, Reshape, MaxPooling2D, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers



size = 128

Pneumonia_model = Sequential()

Pneumonia_model.add(Conv2D(filters = 50, kernel_size = (3,3), padding = 'valid', activation = 'relu', input_shape=(size, size, 3)))
Pneumonia_model.add(MaxPooling2D(pool_size = (2,2), padding = 'valid'))
Pneumonia_model.add(Flatten())
Pneumonia_model.add(Dense(2, activation = 'softmax'))

Pneumonia_model.summary()
Pneumonia_model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
image_size = 150
nb_train_samples = 5216 # number of files in training set
batch_size = 16

EPOCHS = 6
STEPS = nb_train_samples / batch_size

## Specify the values for all arguments to data_generator_with_aug.
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                             horizontal_flip = True,
                                             width_shift_range = 0.2,
                                             height_shift_range = 0.2,
                                             shear_range = 0.2,
                                             zoom_range = 0.2
                                            )
            
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input            
                                          )

train_generator = data_generator_with_aug.flow_from_directory(
       directory = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train/',
       target_size = (image_size, image_size),
       batch_size = batch_size,
       class_mode = 'categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val/',
       target_size = (image_size, image_size),
       class_mode = 'categorical')

test_generator = data_generator_no_aug.flow_from_directory(
       directory = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/',
       target_size = (image_size, image_size),
       batch_size = batch_size,
       class_mode = 'categorical')

#callback = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=7, verbose=0, mode= 'max', baseline= None, restore_best_weights=True)


Pneumonia_model.fit_generator(
       train_generator, # specify where model gets training data
       epochs = 52,
       steps_per_epoch=100,
       validation_data=validation_generator)
       #callbacks = [callback]) # specify where model gets validation data

# Evaluate the model
scores = Pneumonia_model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (Pneumonia_model.metrics_names[1], scores[1]*100))
