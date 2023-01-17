import tensorflow as tf

import numpy as numpy

import os

from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

from PIL import Image

basedir = "/kaggle/input/watermarked-not-watermarked-images/wm-nowm" #here below the train and validation data

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
traindir = os.path.join(basedir,'train') # root for training

validdir = os.path.join(basedir,'valid') # root for testing

traingenerator = ImageDataGenerator(rescale=1./255)

validgenerator = ImageDataGenerator(rescale=1./255)
train_data = traingenerator.flow_from_directory(traindir,target_size=(150,150),batch_size=100,class_mode="binary")

valid_data = validgenerator.flow_from_directory(validdir,target_size=(150,150),batch_size=100,class_mode="binary")
existing = tf.keras.applications.InceptionResNetV2 (input_shape=(150, 150, 3),include_top=False, pooling='max', weights='imagenet')

#for layer in existing.layers:

#  layer.trainable = False

#existing.summary()

last = existing.get_layer("mixed_7a")

last_output = last.output
# Flatten the output layer to 1 dimension

x = tf.keras.layers.Flatten()(last_output)

x = tf.keras.layers.Dropout(0.25)(x) 

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = tf.keras.layers.Dense(128, activation='relu')(x)





x = tf.keras.layers.Dense(64, activation='relu')(x)

# Add a final sigmoid layer for classification

x = tf.keras.layers.Dense  (1, activation='sigmoid')(x)           



model = tf.keras.Model( existing.input, x) 



#model.summary()




model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['accuracy'])
history = model.fit(train_data,

                          validation_data=valid_data,

                          steps_per_epoch=150,

                          epochs=60,

                          validation_steps=50,

                          verbose=2)
model.save_weights('/kaggle/working/latest.h5')
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()