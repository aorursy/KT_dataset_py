import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.image import imread

import cv2

%matplotlib inline
import warnings

warnings.filterwarnings('ignore')
#importing required libraries

from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img ,img_to_array

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report,confusion_matrix
my_data_dir = '../input/real-life-industrial-dataset-of-casting-product/casting_data/'

train_path = my_data_dir + 'train/'

test_path = my_data_dir + 'test/'



image_shape = (300,300,1)

batch_size = 32 #according your model and your choise
# view some images

img = plt.imread('../input/real-life-industrial-dataset-of-casting-product/casting_data/train/def_front/cast_def_0_1001.jpeg')

plt.figure(figsize=(12,8))

plt.imshow(img,cmap='gray')
img1 = plt.imread('../input/real-life-industrial-dataset-of-casting-product/casting_data/train/def_front/cast_def_0_1004.jpeg')

plt.figure(figsize=(12,8))

plt.imshow(img1,cmap='gray')
image_gen = ImageDataGenerator(rescale=1/255)# Rescale the image by normalzing it)

#we using keras inbuild function to ImageDataGenerator so we donnot need to lable all images into 0 and 1 it automatic create it and batch also during trainng 

train_set = image_gen.flow_from_directory(train_path,

                                               target_size=image_shape[:2],

                                                color_mode="grayscale",

                                               batch_size=batch_size,

                                               class_mode='binary',shuffle=True)



test_set = image_gen.flow_from_directory(test_path,

                                               target_size=image_shape[:2],

                                               color_mode="grayscale",

                                               batch_size=batch_size,

                                               class_mode='binary',shuffle=False)
train_set.class_indices
#Creating model



model = Sequential()



model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu',))

model.add(MaxPooling2D(pool_size=(2, 2)))









model.add(Flatten())



model.add(Dense(224))

model.add(Activation('relu'))



# Last layer

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',patience=2)
results = model.fit_generator(train_set,epochs=20,

                              validation_data=test_set,

                             callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()
losses[['accuracy','val_accuracy']].plot()
#first we will find predict probability

pred_probability = model.predict_generator(test_set)
#here it's true label for test set

test_set.classes
predictions = pred_probability > 0.5

#if model predict greater than 0.5 it conveted to 1 means ok_front
print(classification_report(test_set.classes,predictions))
plt.figure(figsize=(10,6))

sns.heatmap(confusion_matrix(test_set.classes,predictions),annot=True)
#we already have string of test path

test_path
img = cv2.imread(test_path+'ok_front/cast_ok_0_1020.jpeg',0)

img = img/255 #rescalinng

pred_img =img.copy()
plt.figure(figsize=(12,8))

plt.imshow(img,cmap='gray')
prediction = model.predict(img.reshape(-1,300,300,1))

if (prediction<0.5):

    print("def_front")

    cv2.putText(pred_img, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:

    print("ok_front")

    cv2.putText(pred_img, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

plt.imshow(pred_img,cmap='gray')

plt.axis('off')

plt.show()
img1 = cv2.imread(test_path+'def_front/cast_def_0_1134.jpeg',0)

img1 = img1/255

pred_img1 =img1.copy()
plt.figure(figsize=(12,8))

plt.imshow(img1,cmap='gray')
model.predict_proba(img.reshape(1,300,300,1))
prediction = model.predict(img1.reshape(-1,300,300,1))

if (prediction<0.5):

    print("def_front")

    cv2.putText(pred_img1, "def_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

else:

    print("ok_front")

    cv2.putText(pred_img1, "ok_front", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

plt.imshow(pred_img1,cmap='gray')

plt.axis('off')

plt.show()
model.save('inspection_of_casting_products.h5')