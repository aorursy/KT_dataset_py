# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import models 
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
def pic(file_path):
    img=mpimg.imread(file_path)
    plt.figure()
    plt.imshow(img)
building=['../input/intel-image-classification/seg_train/seg_train/buildings/10018.jpg',
         '../input/intel-image-classification/seg_train/seg_train/buildings/1001.jpg',
         '../input/intel-image-classification/seg_train/seg_train/buildings/10144.jpg']
for i in building:
    pic(i)
forest=['../input/intel-image-classification/seg_train/seg_train/forest/10037.jpg',
       '../input/intel-image-classification/seg_train/seg_train/forest/10098.jpg',
       '../input/intel-image-classification/seg_train/seg_train/forest/10142.jpg']
for i in forest:
    pic(i)
glacier=['../input/intel-image-classification/seg_train/seg_train/glacier/10024.jpg',
        '../input/intel-image-classification/seg_train/seg_train/glacier/10104.jpg',
        '../input/intel-image-classification/seg_train/seg_train/glacier/1010.jpg']
for i in glacier:
    pic(i)
mountain=['../input/intel-image-classification/seg_train/seg_train/mountain/10031.jpg',
         '../input/intel-image-classification/seg_train/seg_train/mountain/10058.jpg',
         '../input/intel-image-classification/seg_train/seg_train/mountain/10057.jpg']
for i in mountain:
    pic(i)
sea=['../input/intel-image-classification/seg_train/seg_train/sea/10093.jpg',
    '../input/intel-image-classification/seg_train/seg_train/sea/10061.jpg',
    '../input/intel-image-classification/seg_train/seg_train/sea/10114.jpg',
    '../input/intel-image-classification/seg_train/seg_train/sea/10122.jpg']
for i in sea:
    pic(i)
streat=['../input/intel-image-classification/seg_train/seg_train/street/10097.jpg',
       '../input/intel-image-classification/seg_train/seg_train/street/10070.jpg',
       '../input/intel-image-classification/seg_train/seg_train/street/10085.jpg']
for i in streat:
    pic(i)
train_directory='../input/intel-image-classification/seg_train/seg_train/'

val_directory='../input/intel-image-classification/seg_test/seg_test'

test_directory='../input/intel-image-classification/seg_test/seg_test'
train_datagen=ImageDataGenerator(rescale=1/255)

val_datagen=ImageDataGenerator(rescale=1/255)

test_datagen=ImageDataGenerator(rescale=1/255)
train_generator=train_datagen.flow_from_directory(train_directory,target_size=(150,150),
                                                 class_mode='sparse',batch_size=128)

val_generator=val_datagen.flow_from_directory(val_directory,target_size=(150,150),
                                             class_mode='sparse',batch_size=128)

test_generator=test_datagen.flow_from_directory(test_directory,target_size=(150,150),class_mode='sparse')
model_0=models.Sequential()


model_0.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',input_shape=(150,150,3)))
model_0.add(MaxPooling2D())


model_0.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform'))
model_0.add(MaxPooling2D())

model_0.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform'))
#model_0.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform'))
model_0.add(MaxPooling2D())

model_0.add(Conv2D(256,(3,3),activation='relu',kernel_initializer='he_uniform'))
'''model_0.add(Conv2D(256,(3,3),activation='relu',kernel_initializer='he_uniform'))
model_0.add(MaxPooling2D())'''


model_0.add(Flatten())

model_0.add(Dropout(0.25))
model_0.add(Dense(300,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model_0.add(Dense(150,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model_0.add(Dense(40,activation='relu',kernel_initializer='he_uniform'))


model_0.add(Dense(6,activation='softmax'))

model_0.summary()
model_0.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history_0=model_0.fit_generator(train_generator,
                   validation_data=val_generator,
                   epochs=15)
plot_accuracy_loss(history_0)
history_0=model_0.fit_generator(train_generator,
                   validation_data=val_generator,
                   epochs=5)
history_0=model_0.fit_generator(train_generator,
                   validation_data=val_generator,
                   epochs=5)
model_0.evaluate_generator(test_generator)
from keras.applications import VGG16
conv_base=VGG16(input_shape=(150,150,3),include_top=False)
conv_base.summary()
model=models.Sequential()

model.add(conv_base)

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(units=150,kernel_initializer='he_uniform',activation='relu',kernel_regularizer='l2'))
model.add(Dense(units=80,kernel_initializer='he_uniform',activation='relu',kernel_regularizer='l2'))
model.add(Dense(units=30,kernel_initializer='he_uniform',activation='relu',kernel_regularizer='l2'))
model.add(Dense(units=6,activation='softmax'))

model.summary()
model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
history=model.fit_generator(train_generator,epochs=9,validation_data=val_generator)
history=model.fit_generator(train_generator,epochs=3,validation_data=val_generator)
model.evaluate_generator(test_generator)
def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(20,10))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()
plot_accuracy_loss(history)
model.save('model.h5')
labels=['buildings','forest','glacier','mountain','sea','street']
def pred(img):
    img=img_to_array(img)
    img.shape
    img=np.expand_dims(img,[0])
    img=img/255
    img.shape
    print(labels[np.argmax(model.predict(img))])
img=load_img('../input/intel-image-classification/seg_pred/seg_pred/10012.jpg',target_size=(150,150))
img
pred(img)
img=load_img('../input/intel-image-classification/seg_pred/seg_pred/10175.jpg',target_size=(150,150))
img
pred(img)
img=load_img('../input/intel-image-classification/seg_pred/seg_pred/10272.jpg',target_size=(150,150))
img
pred(img)
img=load_img('../input/intel-image-classification/seg_pred/seg_pred/10286.jpg',target_size=(150,150))
img
pred(img)