import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

import os

train_dir='../input/chest-xray-pneumonia/chest_xray/train'
print(len(os.listdir(train_dir)))
test_dir='../input/chest-xray-pneumonia/chest_xray/test'
print(len(os.listdir(test_dir)))
import matplotlib.image as mpimg

plt.figure(figsize=(10,10))
img = mpimg.imread('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0115-0001.jpeg')
plt.imshow(img,cmap='gray')
print('NORMAL CHEST')
plt.figure(figsize=(10,10))
img = mpimg.imread('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1003_virus_1685.jpeg')
plt.imshow(img,cmap='gray')
print('PNEUMONIA CHEST')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1.0/255.0)

trainDatagen=datagen.flow_from_directory(train_dir,
                                 target_size=(250,250),
                                 batch_size=50,
                                 class_mode='binary',
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True)
valDatagen=datagen.flow_from_directory(test_dir,
                                      target_size=(250,250),
                                      batch_size=10,
                                      class_mode='binary')
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(250,250, 3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
    
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
    
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.3),
    
                                    tf.keras.layers.Flatten(),
    
                                    tf.keras.layers.Dense(512, activation='relu'),
    
                                    tf.keras.layers.Dense(1, activation='sigmoid')])


model.summary()
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

class mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epochs,logs={}):
        if(logs.get('accuracy')>0.99):
            self.model.stop_training=True
callbacks=mycallbacks()
history=model.fit_generator(trainDatagen,validation_data=valDatagen,epochs=30,verbose=1,callbacks=[callbacks])
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
