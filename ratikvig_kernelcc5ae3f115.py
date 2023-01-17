# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

"""

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

"""
import matplotlib.pyplot as plt

import cv2
DATADIR = "../input/aptos2019-blindness-detection/train_images/"

df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')




training_data = []

""""for img in df['id_code']:

    path = os.path.join(DATADIR,img)

    path = path+".png"

    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

   

    #image = image.resize(1024,1024)

    image = cv2.resize(image,(1024,1024))

    print(type(image))

    img_arr.append(image)

    i+=1

    break"""

for _,row in df.iterrows():

    print (row['id_code'],' ',row['diagnosis'])

    path = os.path.join(DATADIR,row['id_code']+".png")

    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image,(224,224))

    training_data.append([image,row['diagnosis']])

    

del df



    

df = pd.read_csv('../input/resized-2015-2019-blindness-detection-images/labels/trainLabels15.csv')

DATADIR2 = '../input/resized-2015-2019-blindness-detection-images/resized train 15/'

for _,row in df.iterrows():

    print (row['image'],' ',row['level'])

    path = os.path.join(DATADIR2,row['image']+".jpg")

    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image,(224,224))

    training_data.append([image,row['level']])

   

    

count=0

for i in range(35126):

    if(df['level'][i] == 0):

        count +=1

print(count)
del df

import gc

gc.collect()


df = pd.read_csv('../input/resized-2015-2019-blindness-detection-images/labels/testLabels15.csv')

DATADIR2 = '../input/resized-2015-2019-blindness-detection-images/resized test 15/'

for _,row in df.iterrows():

    print (row['image'],' ',row['level'])

    path = os.path.join(DATADIR2,row['image']+".jpg")

    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image,(224,224))

    training_data.append([image,row['level']])
del df

print(len(training_data))
"""img_arr = np.array(img_arr)

plt.imshow(img_arr[0])

plt.show()

print(img_arr.shape)"""



import keras

x_train=[]

y_train=[]

for features,labels in training_data:

    x_train.append(features)

    y_train.append(labels)

x_train = np.array(x_train).reshape(-1,224,224,1)



y_train = np.array(y_train)

y_train = keras.utils.to_categorical(y_train)



print(y_train.shape)
training_data = []
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D,AveragePooling2D,BatchNormalization

import tensorflow as tf



input_shape = (224, 224, 1)

model = Sequential([

Conv2D(64, (3, 3), input_shape=(224, 224, 1), activation='relu',padding='same'),

Conv2D(64, (3, 3), activation='relu',padding='same'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(128, (3, 3), activation='relu',padding='same'),

Conv2D(128, (3, 3), activation='relu',padding='same'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(256, (3, 3), activation='relu',padding='same'),

Conv2D(256, (3, 3), activation='relu',padding='same'),

Conv2D(256, (3, 3), activation='relu',padding='same'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(512, (3, 3), activation='relu',padding='same'),

Conv2D(512, (3, 3), activation='relu',padding='same'),

Conv2D(512, (3, 3), activation='relu',padding='same'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Conv2D(512, (3, 3), activation='relu',padding='same'),

Conv2D(512, (3, 3), activation='relu',padding='same'),

Conv2D(512, (3, 3), activation='relu',padding='same'),

MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

Flatten(),

Dense(4096, activation='relu'),

Dense(4096, activation='relu'),

Dense(5, activation='softmax')

])

model.summary()

"""

model = keras.Sequential()



model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(224,224,1)))

model.add(AveragePooling2D())



model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

model.add(AveragePooling2D())



model.add(Flatten())



model.add(Dense(units=120, activation='relu'))



model.add(Dense(units=84, activation='relu'))



model.add(Dense(units=5, activation = 'softmax'))

model.summary()







model = Sequential()





model.add(Conv2D(filters=96, input_shape=(224,224,1), kernel_size=(11,11),\

 strides=(4,4), padding='valid'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(1024, input_shape=(224*224*1,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())





model.add(Dense(5))

model.add(Activation('softmax'))



model.summary()"""

print(y_train)
import keras

opt = keras.optimizers.Adam(lr=0.00001)

class_weights = {0:1.,1:9.,2:4.,3:26.,4:26.}

model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=1,validation_split=0.1,class_weight = class_weights)
model.save_weights('my_model.h5')
test_data = []

TEST_DIR = "../input/aptos2019-blindness-detection/test_images/"

df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

for _,row in df.iterrows():

    print (row['id_code'])

    path = os.path.join(TEST_DIR,row['id_code']+".png")

    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image,(224,224))

    test_data.append(image)
test_data = np.array(test_data).reshape(-1,224,224,1)

res = model.predict(test_data)
diagnosis = np.argmax(res,axis=1)

my_submission = pd.DataFrame({'id_code': df.id_code, 'diagnosis': diagnosis})



my_submission.to_csv('submission.csv', index=False)