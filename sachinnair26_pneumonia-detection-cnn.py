train_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/train"
test_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"
val_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/val"
import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten, Dense, Activation,BatchNormalization,Dropout
train_generator = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_set = train_generator.flow_from_directory(train_dir,target_size = (200, 200),
  batch_size = 32,
   class_mode = 'binary')
x_batch, y_batch = next(train_set)
fig, (ax1,ax2) = plt.subplots(1,2)
image1 = x_batch[0]
image2 = x_batch[1]
ax1.imshow(image1)
ax2.imshow(image2)
plt.show()
val_generator = ImageDataGenerator(rescale = 1./255)
val_set = val_generator.flow_from_directory(val_dir,target_size=(200,200),batch_size=32,class_mode='binary')
test_generator = ImageDataGenerator(rescale = 1./255)
test_set = test_generator.flow_from_directory(test_dir,target_size=(200,200),batch_size=32,class_mode='binary')
model = Sequential()

# model.add(Conv2D(256,(3,3),(1,1),padding='valid'))
# model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=(1,1)))
# model.add(Activation('relu'))

model.add(Conv2D(128,(3,3),(1,1),padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=(1,1)))
model.add(Activation('relu'))


model.add(Conv2D(64,(3,3),padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(32,(3,3),(1,1),padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(16,(5,5),(1,1),padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=(1,1)))
model.add(Activation('relu'))

model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(256,'relu'))
# model.add(Dropout(0.6))
model.add(Dense(128,'relu'))
# model.add(Dropout(0.2))

model.add(Dense(64,'relu'))
model.add(Dense(1,'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_set,validation_data=val_set,batch_size=32,epochs=10)
score = model.evaluate(test_set)
print(score)  
