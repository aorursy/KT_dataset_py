from keras.preprocessing.image import ImageDataGenerator, load_img

import os
img=load_img('../input/chest_xray/chest_xray/test/PNEUMONIA/person147_bacteria_706.jpeg')
img
total_train_normal=os.listdir('../input/chest_xray/chest_xray/train/NORMAL/')
total_train_pneumonia=os.listdir('../input/chest_xray/chest_xray/train/PNEUMONIA/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as rd
import cv2
normal_xr=rd.sample(total_train_normal,10)
f,ax = plt.subplots(2,5, figsize=(30,10))
for i in range(0,10):
    img=cv2.imread('../input/chest_xray/chest_xray/train/NORMAL/'+normal_xr[i])
    ax[i//5,i%5].imshow(img)
    ax[i//5,i%5].axis('off')
f.suptitle('Normal Lungs')
plt.show()
pneumonia_xr = rd.sample(total_train_pneumonia,10)
f,ax = plt.subplots(2,5, figsize=(30,10))

for i in range(0,10):
    img_2 = cv2.imread('../input/chest_xray/chest_xray/train/PNEUMONIA/'+pneumonia_xr[i])
    ax[i//5,i%5].imshow(img_2)
    ax[i//5,i%5].axis('off')
f.suptitle('Pneumonia Lungs')
plt.show()
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
model=Sequential()
image_width=150
image_height=150
batch_size=20
no_of_epoch=10

model.add(Conv2D(32,(3,3),input_shape=(image_height,image_width,3),activation='relu'))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())



model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1/255)
training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',
                                                 target_size=(image_width, image_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')
test_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',
                                                 target_size=(image_width, image_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)
callbacks=reduce_learning_rate
callbacks
Result=model.fit_generator(training_set,
                    steps_per_epoch=5216//batch_size,
                    epochs=2,
                    validation_data=test_set,
                    validation_steps=624//batch_size,
                    
                   )
