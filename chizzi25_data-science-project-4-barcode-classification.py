from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential

import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
num_classes = 1



my_model = Sequential()

my_model.add(ResNet50(include_top = False, weights = 'imagenet',pooling = 'avg'))

my_model.add(Dense(num_classes, activation = 'sigmoid'))



my_model.layers[0].trainable = False



my_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
my_model.summary()
image_size = 224



datagenerator = ImageDataGenerator(preprocessing_function = preprocess_input,

                                   horizontal_flip = True, width_shift_range = 0.2, height_shift_range = 0.2)



train_generator = datagenerator.flow_from_directory('/kaggle/input/barcode-test-set/training_set',

                                                   target_size = (image_size, image_size),

                                                   batch_size = 24,

                                                   class_mode = 'binary')



val_generator = datagenerator.flow_from_directory('/kaggle/input/barcode-test-set/test_set',

                                                 target_size = (image_size, image_size),

                                                 class_mode = 'binary')



my_model.fit(train_generator, steps_per_epoch = 7, 

          validation_data = val_generator, validation_steps = 1, epochs = 1)
val_generator.class_indices
preds = my_model.predict(val_generator)

labels = np.argmax(preds, axis = 1)

for i in labels:

    if i == 0:

        print('The model predicted the image as a barcode')

    else:

        pass