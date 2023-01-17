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

traingenerator = ImageDataGenerator(rescale=.1/255)
validgenerator = ImageDataGenerator(rescale=.1/255)
train_data = traingenerator.flow_from_directory(traindir,target_size=(150,150),batch_size=20,class_mode="binary")
valid_data = validgenerator.flow_from_directory(validdir,target_size=(150,150),batch_size=20,class_mode="binary")
model = tf.keras.Sequential([
    #tf.keras.layers.Conv2D(16,(1,1), activation='relu', input_shape=(150, 150, 3)),
    #tf.keras.layers.MaxPool2D(2,2),
    #tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    #tf.keras.layers.MaxPool2D(2,2),
    #tf.keras.layers.Conv2D(64,(3,34), activation='relu'),
    #tf.keras.layers.MaxPool2D(2,2),
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(128, activation='elu'),
    #tf.keras.layers.Dense(64, activation='elu'),
    #tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.applications.ResNet50(input_shape=(150, 150, 3),include_top=False, pooling='max', weights='imagenet'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1,activation="sigmoid")

])
model.layers[0].trainable=True
model.summary()


model.compile(optimizer='adam',#RMSprop(lr=0.001)
              loss='binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(train_data,
                              validation_data=valid_data,
                              steps_per_epoch=100,
                              epochs=200,
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