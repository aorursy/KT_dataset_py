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
categories=list(os.listdir('/kaggle/input/cat-and-dog/training_set/training_set'))
dire='/kaggle/input/cat-and-dog/training_set/training_set'
#first preprocess the training data 

import cv2

import matplotlib.pyplot as plt

features=[]

IMG_SIZE=100

for i in categories:

    path=os.path.join(dire,i)

    num_classes=categories.index(i)

    for img in os.listdir(path):

        if img.endswith('.jpg'):

            

            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

            img_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))

            features.append([img_array,num_classes])
#lets create the dependent and independent variable 

x=[]

y=[]

for i,j in features:

    x.append(i)

    y.append(j)
#lets visualize the training data

for i in range(1,5):

    plt.imshow(x[i])

    plt.xlabel(y[i])

    plt.show()
#lets reshape the size of x to meet the keras requirement

x=np.array(x).reshape(-1,100,100,1)
x.shape
#one hot encoding for the target lables

from tensorflow.keras.utils import to_categorical

y=to_categorical(y)
#now split the data into training and test sets

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
#now import the necessary modules require to build the model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout,MaxPool2D
model=Sequential([

    Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(100,100,1)),

    MaxPooling2D((2,2)),

    Dropout(0.2),

    BatchNormalization(),

    Conv2D(128,(3,3),activation='relu',padding='same'),

    MaxPooling2D((2,2)),

    Dropout(0.25),

    BatchNormalization(),

    Conv2D(256,(3,3),activation='relu',padding='same'),

    MaxPooling2D((2,2)),

    Dropout(0.3),

    BatchNormalization(),

    Flatten(),

    Dense(2,activation='sigmoid')

])
model.summary()
model.compile(optimizer='Adam',loss='mae',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=15,validation_split=0.30,batch_size=50)
#lets take a look on the loss on both the training set and test set

plt.figure(figsize=(12,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('mean_absolute_error')

plt.title('epochs-vs-loss')

plt.legend(['train_set','test_set'])

plt.show()
#lets analyse the accuracy score on both train and test sets

plt.figure(figsize=(12,5))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.xlabel('epochs')

plt.ylabel('accuracy score')

plt.title('epochs-vs-accuracy score')

plt.legend(['train_set','test_set'])

plt.show()
#clearly the model is overfitting we have to apply data augmentation to overcome the problem of overfitting.
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_dir='/kaggle/input/cat-and-dog/training_set/training_set'

test_dir='/kaggle/input/cat-and-dog/test_set/test_set'

train_datagen=ImageDataGenerator(rescale=1./255,

                                rotation_range=40,

                                shear_range=0.2,

                                width_shift_range=0.2,

                                height_shift_range=0.2,

                                horizontal_flip=True,

                                zoom_range=0.2,

                                fill_mode='nearest')

test_datagen=ImageDataGenerator(rescale=1./255)



train_generator=train_datagen.flow_from_directory(train_dir,

                                                 target_size=(150,150),

                                                 batch_size=64,

                                                 class_mode='binary')

test_generator=test_datagen.flow_from_directory(test_dir,

                                               target_size=(150,150),

                                               batch_size=50,

                                               class_mode='binary')


model_1= Sequential([

    Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),

    MaxPool2D(2,2),

    Conv2D(32,(3,3),activation='relu'),

    Conv2D(32,(3,3),activation='relu'),

    MaxPool2D(2,2),

    Conv2D(64,(3,3),activation='relu'),

    MaxPool2D(2,2),

    Dropout(0.3),

    Flatten(),

    Dense(256,activation='relu'),

    Dense(128,activation='relu'),

    Dense(64,activation='relu'),

    Dropout(0.25),

    Dense(1,activation='sigmoid')

])



model_1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
history_1= model_1.fit(train_generator,

                       validation_data=test_generator,

                              epochs=30,

                              steps_per_epoch=125

                              )
#lets have a look at loss 

plt.figure(figsize=(12,5))

plt.plot(history_1.history['loss'])

plt.plot(history_1.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('mean_absolute_error')

plt.title('epochs-vs-loss')

plt.legend(['train_set','test_set'])

plt.show()
plt.figure(figsize=(12,5))

plt.plot(history_1.history['accuracy'])

plt.plot(history_1.history['val_accuracy'])

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.title('epochs-vs-accuracy')

plt.legend(['train_set','test_set'])

plt.show()
#now the model is performing quite well on the test set