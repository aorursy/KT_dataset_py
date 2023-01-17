# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_dir='/kaggle/input/intel-image-classification/seg_train/seg_train'

test_dir='/kaggle/input/intel-image-classification/seg_test/seg_test'
#lets view the no of categories in the directory

categories=os.listdir(train_dir)
#preprocessing the data to load into an list

import cv2

import matplotlib.pyplot as plt

x_1=[]

for i in categories:

    path=os.path.join(train_dir,i)

    class_num=categories.index(i) #creates labels for the classification

    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

        img_array=cv2.resize(img_array,(100,100))

        x_1.append([img_array,class_num])
#separating the features and labels into two separate lists

x=[]

y=[]

for i ,j in x_1:

    x.append(i)

    y.append(j)
#lets reshape the data to meet the keras requirements

x=np.array(x).reshape(-1,100,100,1)
x=x/255
#one hot-encoding for the target lables with 6 classes

from tensorflow.keras.utils import to_categorical

y=to_categorical(y)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
#importing the necessary libraries

from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Dropout,Conv2D,MaxPool2D

from tensorflow.keras.models import Sequential

from tensorflow.keras import regularizers

from tensorflow.keras import initializers

model=Sequential([

    Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(100,100,1)),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(64,(3,3),padding='same',activation='relu'),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(128,(3,3),padding='same',activation='relu'),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(256,(3,3),padding='same',activation='relu'),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(512,(3,3),padding='same',activation='relu'),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Flatten(),

    Dense(1024,activation='relu'),

    Dropout(0.25),

    BatchNormalization(),

    Dense(6,activation='softmax')

    

])
model.compile(optimizer="Adam", loss="mse", metrics=["acc"])
history=model.fit(x_train,y_train,epochs=30,batch_size=50,validation_split=0.30)
loss,accuracy=model.evaluate(x_test,y_test)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.title('model accuracy')

plt.legend(['train_data','validation_data'])

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(['train_data','test_data'])

plt.show()
#the above model seems to be overfitting with some sharp spikes, so to overcome it we need to perform data augmentation
#new model for the augmentation

model_1=Sequential([

    Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(100,100,3)),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(64,(3,3),padding='same',activation='relu'),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(128,(3,3),padding='same',activation='relu'),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(256,(3,3),padding='same',activation='relu'),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(512,(3,3),padding='same',activation='relu'),

    MaxPool2D((2,2),strides=(2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Flatten(),

    Dense(1024,activation='relu'),

    Dropout(0.25),

    BatchNormalization(),

    Dense(6,activation='softmax')

    

])
model_1.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=["acc"])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,width_shift_range=.15,height_shift_range=0.2,zoom_range=0.2,shear_range=0.2,rotation_range=90,horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(

batch_size=50,

directory=train_dir,

shuffle=True,

target_size=(100, 100)   

)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(

batch_size=50,

directory=test_dir,

shuffle=True,

target_size=(100, 100)   

)
history_1=model_1.fit_generator(train_generator,

                           epochs=40,

                           validation_data=test_generator,

                           verbose=2     

                           )
plt.figure(figsize=(12,5))

plt.plot(history_1.history['loss'])

plt.plot(history_1.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(['train_data','test_data'])

plt.title('loss analysis')

plt.show()
plt.figure(figsize=(12,5))

plt.plot(history_1.history['acc'])

plt.plot(history_1.history['val_acc'])

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend(['train_data','test_data'])

plt.title('accuracy analysis')

plt.show()
#performing slighly better than the previos model with less no of sharp spikes