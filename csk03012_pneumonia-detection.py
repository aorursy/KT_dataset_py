import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

from keras.preprocessing import image



model = tf.keras.models.Sequential([

  

    # Note the input shape is the desired size of the image 300x300 with 3 bytes color

    # This is the first convolution

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

  

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

  

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

  

    # The fourth convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

  

    # The fifth convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



  

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'), # 512 neuron hidden layer

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('normal') clas and 1 for ('pneumonia') class

    tf.keras.layers.Dense(1, activation='sigmoid')

])



# to get the summary of the model

model.summary()



# configure the model for traning by adding metrics

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1/255)

test_datagen = ImageDataGenerator(rescale = 1/255)



train_generator = train_datagen.flow_from_directory(

    '../input/chest-xray-pneumonia/chest_xray/train',

    target_size = (300,300),

    batch_size = 128,

    class_mode = 'binary'

)



validation_generator = test_datagen.flow_from_directory(

    '../input/chest-xray-pneumonia/chest_xray/test',

    target_size = (300, 300),

    batch_size = 128,

    class_mode = 'binary'

)
# training the model

history = model.fit(

    train_generator,

    steps_per_epoch = 10,

    epochs = 10,

    validation_data = validation_generator

)
# load new unseen dataset

eval_datagen = ImageDataGenerator(rescale = 1/255)



test_generator = eval_datagen.flow_from_directory(

    '../input/chest-xray-pneumonia/chest_xray/val',

    target_size = (300, 300),

    batch_size = 128, 

    class_mode = 'binary'

)



eval_result = model.evaluate_generator(test_generator, 624)

print('loss rate at evaluation data :', eval_result[0])

print('accuracy rate at evaluation data :', eval_result[1])