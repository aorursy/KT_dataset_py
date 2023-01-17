import os

for dirname, _, filenames in os.walk('/kaggle/input/fruit-images-for-object-detection/train_zip/train'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# 필요한 모듈 임포트

import pandas as pd

import numpy as np

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

import os

import cv2

import matplotlib.pyplot as plt

from keras import optimizers

from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers import Dense,Activation,AveragePooling2D,MaxPooling2D,Dropout,BatchNormalization
# train을 train-val으로 나누기 위한 모듈 임포트

from sklearn.model_selection import train_test_split



np.random.seed(42)
# train set 불러와서 reshape 수행 후 train-val set 으로 나누기

# matplotlib은 RGB, 저장된 이미지는 BGR순이므로 cv2모듈을 이용해 바꿔주기

# 그래야 파란색 이미지로 안나옴

# 

from keras.preprocessing.image import ImageDataGenerator



train_images = []       

train_labels = []

shape = (200,200)  

train_path = '../input/fruit-images-for-object-detection/train_zip/train'



for filename in os.listdir('../input/fruit-images-for-object-detection/train_zip/train'):

    if filename.split('.')[1] == 'jpg':

        img = cv2.imread(os.path.join(train_path,filename))

        

        # Spliting file names and storing the labels for image in list

        train_labels.append(filename.split('_')[0])

        

        # Resize all images to a specific shape

        img = cv2.resize(img,shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        train_images.append(img)



# Converting labels into One Hot encoded sparse matrix

train_labels = pd.get_dummies(train_labels).values



# Converting train_images to array

x_train = np.array(train_images)





# # Splitting Training data into train and validation dataset

x_train,x_val,y_train,y_val = train_test_split(x_train,train_labels,random_state=42)
print(len(x_train),len(x_val))
print(x_train.shape)
print(train_labels.shape)
print(y_train.shape)
# to increase the number of train images

datagen = ImageDataGenerator(rotation_range=20,

                                           width_shift_range=0.05,

                                           height_shift_range=0.05)

datagen.fit(x_train)

train_inc = datagen.flow(x_train, y_train,

                                    batch_size=16)
# test set 도 동일한 과정 거치기

test_images = []

test_labels = []

shape = (200,200)

test_path = '../input/fruit-images-for-object-detection/test_zip/test'



for filename in os.listdir('../input/fruit-images-for-object-detection/test_zip/test'):

    if filename.split('.')[1] == 'jpg':

        img = cv2.imread(os.path.join(test_path,filename))

        

        # Spliting file names and storing the labels for image in list

        test_labels.append(filename.split('_')[0])

        

        # Resize all images to a specific shape

        img = cv2.resize(img,shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        test_images.append(img)

        

# Converting test_images to array

x_test = np.array(test_images)
print(x_test.shape)
# 모델 구축

model= Sequential()

model.add(Conv2D(kernel_size=(3,3), filters=32, activation='relu', input_shape=(200,200,3,)))



model.add(Conv2D(filters=32,kernel_size = (3,3),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.4))



model.add(Conv2D(filters=64,kernel_size = (3,3),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.4))



model.add(Conv2D(filters=64,kernel_size = (3,3),activation='relu'))



model.add(Flatten())



model.add(Dense(32,activation='relu'))

model.add(Dense(16,activation='relu'))

model.add(Dense(4,activation = 'softmax'))

    

model.compile(

              loss='categorical_crossentropy', 

              metrics=['acc'],

              optimizer='adam'

             )
model.summary()
# 모델 훈련

# history = model.fit(x_train,train_labels,epochs=50,batch_size=50)

history=model.fit_generator(train_inc,epochs=70,steps_per_epoch=10,

                           verbose=1,validation_data=(x_val,y_val))
# Testing predictions and the actual label

checkImage = x_test

checklabel = test_labels



predict = model.predict(checkImage)

predict_arr=[]



output = { 0:'apple',1:'banana',2:'mixed',3:'orange'}

for i in range(len(x_test)):

    predict_arr.append(output[np.argmax(predict[i])])
#test 정확도

print(checklabel)

print("\n")

print(predict_arr)
# test 정확도 수치화

print(np.mean(np.array(checklabel)==np.array(predict_arr)))