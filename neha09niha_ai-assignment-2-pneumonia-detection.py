# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/chest_xray/chest_xray"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



#instantiate Sequential

my_cnn=Sequential()



#add convolution layer

my_cnn.add(Conv2D(32,(3,3),input_shape=(128,128,3),activation='relu'))



#Add max pooling layer

my_cnn.add(MaxPooling2D(pool_size=(2,2)))



#add another convolution layer

my_cnn.add(Conv2D(32,(3,3),activation='relu'))



#Add max pooling layer

my_cnn.add(MaxPooling2D(pool_size=(2,2)))



#add flatten 

my_cnn.add(Flatten())



#add hidde layer

my_cnn.add(Dense(units=128,activation='relu'))



#add output layer

my_cnn.add(Dense(units=1,activation='sigmoid'))



#compile the model

my_cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



#import this package to do preprocessing of data

from keras.preprocessing.image import ImageDataGenerator



#Before doing test train split create a train model

train_model=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)



#image we need to do flip and do other things while training not testing soo only rescale is enough for test

test_model=ImageDataGenerator(rescale=1./255)



#assign data to train model

train_set=train_model.flow_from_directory('../input/chest_xray/chest_xray/train/',target_size=(128,128),batch_size=32,class_mode='binary')



#assign data to test model

test_set=test_model.flow_from_directory('../input/chest_xray/chest_xray/test/',target_size=(128,128),batch_size=32,class_mode='binary')



#fit the model

my_cnn.fit_generator(train_set,steps_per_epoch=5216/32,epochs=5,validation_data=test_set,validation_steps=624/32)

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator



import numpy as np

from keras.preprocessing import image

test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1950_bacteria_4881.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1951_bacteria_4882.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1952_bacteria_4883.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1947_bacteria_4876.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1949_bacteria_4880.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg',target_size=(128,128))



#NORMAL

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1440-0001.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1437-0001.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1436-0001.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1438-0001.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1442-0001.jpeg',target_size=(128,128))

#test_image=image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg',target_size=(128,128))



test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=my_cnn.predict(test_image)

train_set.class_indices

if result[0][0] == 1.0:

    prediction = 'PNEMONIA'

    print(" The Patient's Lung is affected with ")

    print(prediction)

elif result[0][0] == 0:

    prediction = 'NORMAL'

    print(" The Patient's Lung is ")

    print(prediction)