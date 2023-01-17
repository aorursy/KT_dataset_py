import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as im

import keras

from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,Activation

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator
import cv2

img = im.imread('/kaggle/input/animals10/raw-img/gatto/123.jpeg')

print(img.shape)

plt.imshow(img)

img = cv2.resize(img,(75,75))

plt.imshow(img)
img.shape
import os

os.listdir('/kaggle/input/animals10/raw-img')

#classes
target_size=(75,75)

batch_size=1000

data_generator = ImageDataGenerator(rescale=1/255,validation_split=0.25)

train_data = data_generator.flow_from_directory('/kaggle/input/animals10/raw-img',subset='training',target_size=target_size,batch_size=batch_size,class_mode='categorical',color_mode='grayscale')

test_data = data_generator.flow_from_directory('/kaggle/input/animals10/raw-img',subset='validation',target_size=target_size,batch_size=batch_size,class_mode='categorical',color_mode='grayscale')
count  = 0 



for i in train_data:

    for j in i[0]:

        imk = j

        break

           

    break

img    

#plt.imshow(imk)
''''model = Sequential()

model.add(Conv2D(32,(5,5),input_shape=(75,75,1),activation = 'relu'))

model.add(MaxPooling2D(3,3))



model.add(Conv2D(32,(5,5),activation = 'relu'))

model.add(MaxPooling2D(3,3))



model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

'''

#Alexnet model

model = Sequential()

model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = (75, 75,1)))

model.add(Dropout(0.4))

model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))

model.add(Dropout(0.4))

model.add(MaxPooling2D((2, 2)))

model

model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))

model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))

model.add(Dropout(0.4))

model.add(MaxPooling2D((2, 2)))



model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))

model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))

model.add(Dropout(0.4))

model.add(MaxPooling2D((2, 2)))



model.add(Flatten())

model.add(Dense(10, activation = "softmax"))



#model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])

mod=model.fit_generator(train_data,validation_data=test_data,shuffle=True,epochs=50,steps_per_epoch=19638 //batch_size,validation_steps=6541//batch_size,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=4)])


mod.history
epoch = len(mod.history['val_loss'])



plt.plot(range(1,epoch+1),mod.history['loss'],label='loss')

plt.plot(range(1,epoch+1),mod.history['val_loss'],label='val_loss')

plt.legend()

plt.show()
plt.plot(range(1,epoch+1),mod.history['accuracy'],label='accuracy')

plt.plot(range(1,epoch+1),mod.history['val_accuracy'],label='val_accuracy')

plt.legend()

plt.show()