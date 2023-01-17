# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/plant-seedlings-classification/test'):

    for filename in filenames:

        dirname

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

import os

count=1;

all_images=[]

all_class=[];

path='/kaggle/input/plant-seedlings-classification/train/'

entries = os.listdir('/kaggle/input/plant-seedlings-classification/train/')

for entry in entries:

  for image_path in os.listdir(path+entry):

    img = cv2.imread(path+entry+'/'+image_path)

    img=cv2.resize(img,(32,32))

    GREEN_MIN = np.array([25, 52, 72],np.uint8)

    GREEN_MAX = np.array([102, 255, 255],np.uint8)

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    frame_threshed = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)

    all_images.append(frame_threshed)

    all_class.append(entry)



    



 

X_train=np.array(all_images)

y_train=np.array(all_class)
X_train.shape
train_images=[]

all_class=[]

image_name=[]

path='/kaggle/input/plant-seedlings-classification/test/'





for image_path in os.listdir(path):

  img = cv2.imread(path+image_path)

  img=cv2.resize(img,(32,32))

  GREEN_MIN = np.array([25, 52, 72],np.uint8)

  GREEN_MAX = np.array([102, 255, 255],np.uint8)

  hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

  frame_threshed = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)

  train_images.append(frame_threshed)

  all_class.append(entry)

  image_name.append(image_path)



    



 

x_test=np.array(train_images)


all_class=[]

path='/kaggle/input/plant-seedlings-classification/train/'

entries = os.listdir('/kaggle/input/plant-seedlings-classification/train/')

for entry in entries:

  print(entry)
y_train=[]

train_classes=[]

path='/kaggle/input/plant-seedlings-classification/train/'

entries = os.listdir('/kaggle/input/plant-seedlings-classification/train/')

for entry in entries:

  for image_path in os.listdir(path+entry):

    train_classes.append(entry)

    

y_train=np.array(train_classes)
y_train=np.where(y_train=='Common Chickweed',11, y_train) 

y_train=np.where(y_train=='Charlock', 3, y_train) 

y_train=np.where(y_train=='Shepherds Purse', 1, y_train) 

y_train=np.where(y_train=='Black-grass', 8, y_train) 

y_train=np.where(y_train=='Cleavers', 7, y_train) 

y_train=np.where(y_train=='Scentless Mayweed', 10, y_train) 

y_train=np.where(y_train=='Common wheat', 6, y_train) 

y_train=np.where(y_train=='Fat Hen', 2, y_train) 

y_train=np.where(y_train=='Maize', 5, y_train) 



y_train=np.where(y_train=='Loose Silky-bent', 0, y_train) 

y_train=np.where(y_train=='Sugar beet', 4, y_train) 

y_train=np.where(y_train=='Small-flowered Cranesbill', 9, y_train) 
from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras.utils import np_utils

import tensorflow as tf

from keras.callbacks import EarlyStopping

from keras.optimizers import SGD
trainY=np_utils.to_categorical(y_train,12)
trainX =X_train/255

testX=x_test/255
trainX=trainX.reshape(trainX.shape[0],32,32,1).astype('float32') 

testX=testX.reshape(testX.shape[0],32,32,1).astype('float32') 
trainX.shape
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1), padding='valid', activation='relu'))

model.add(MaxPooling2D(pool_size=2 , padding='same'))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(12, activation='softmax'))

# Compile model

epochs = 40

lrate = 0.01

decay = lrate/epochs

          

#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] , )

model.summary()


epochs=5

early_stopping = EarlyStopping(monitor='acc', patience=2, verbose=1, mode='auto')

callback_list = [early_stopping]# [stats, early_stopping]



model.fit(trainX, trainY,epochs=epochs, batch_size=32 , callbacks=callback_list)
pred_class=model.predict_classes(testX)
pred_class
pred_class=np.where(pred_class=='11', 'Common Chickweed',pred_class) 

pred_class=np.where(pred_class== '3','Charlock', pred_class) 

pred_class=np.where(pred_class== '2','Shepherds Purse', pred_class) 

pred_class=np.where(pred_class== '8','Black-grass', pred_class) 

pred_class=np.where(pred_class== '7', 'Cleavers',pred_class) 

pred_class=np.where(pred_class== '10','Scentless Mayweed', pred_class) 

pred_class=np.where(pred_class== '6', 'Common wheat',pred_class) 

pred_class=np.where(pred_class== '2', 'Fat Hen',pred_class) 

pred_class=np.where(pred_class=='5', 'Maize', pred_class) 

pred_class=np.where(pred_class== '0', 'Loose Silky-bent',pred_class) 

pred_class=np.where(pred_class== '4', 'Sugar beet',pred_class) 

pred_class=np.where(pred_class== '9', 'Small-flowered Cranesbill',pred_class) 

pred_class
cnt=0

result=[]

df_result=pd.DataFrame()

for x in pred_class:

    result.append(image_name[cnt]+","+pred_class[cnt])

    

    cnt=cnt+1

import pandas as pd

df=pd.DataFrame(result)
df_result1=pd.DataFrame()



df_result1['file']=image_name

df_result1['species']=pred_class
result
df_result1
df_result1.to_excel('sample_submission.xlsx',index=False)