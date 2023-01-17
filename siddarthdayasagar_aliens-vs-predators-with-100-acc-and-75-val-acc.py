import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
train_dir='../input/alien-vs-predator-images/data/train'
validation_dir='../input/alien-vs-predator-images/data/validation'
import os

print(os.listdir(train_dir))

print(os.listdir(validation_dir))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1.0/255.0)

trainDatagen=datagen.flow_from_directory(train_dir,
                                 target_size=(250,250),
                                 batch_size=50,
                                 class_mode='binary')
valDatagen=datagen.flow_from_directory(validation_dir,
                                      target_size=(250,250),
                                      batch_size=10,
                                      class_mode='binary')

from tensorflow.keras import regularizers
independenceday=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='tanh',input_shape=(250,250,3)),
                                           tf.keras.layers.MaxPooling2D(2,2),
                                           tf.keras.layers.Dropout(0.2),
                                           tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                                           tf.keras.layers.MaxPooling2D(2,2),
                                           
                                           tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                           tf.keras.layers.MaxPooling2D(2,2),
                                           tf.keras.layers.Dropout(0.2),
                                           tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                           tf.keras.layers.MaxPooling2D(2,2),
                                           tf.keras.layers.Dropout(0.5),
                                           tf.keras.layers.Flatten(),
                                           tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.001)),
                                            tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.001)),
                                           tf.keras.layers.Dense(1,activation='sigmoid')])
independenceday.summary()
independenceday.compile(loss='binary_crossentropy',
                       optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                       metrics=['accuracy'])
history=independenceday.fit_generator(trainDatagen,validation_data=valDatagen,epochs=150,verbose=1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

