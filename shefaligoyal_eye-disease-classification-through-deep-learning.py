# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import cv2

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Flatten

from tensorflow.keras.layers import Conv2D,MaxPool2D

from tensorflow.keras.layers import Input, Dense

from keras.utils import to_categorical

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt



from tensorflow.keras.applications.vgg16 import VGG16

import pandas as pd

import numpy as np

import os

from numpy import argmax

PATH = "/kaggle/input/eye-disease-dataset/Eye_diseases"

data_dir_list = os.listdir(PATH)

data_dir_list
img_rows=224

img_cols=224

num_channel=3



num_epoch = 50

batch_size = 32



img_data_list=[]

classes_names_list=[]

target_column=[]

for dataset in data_dir_list:

    classes_names_list.append(dataset)

    print("Getting image from {} folder".format(dataset))

    img_list= os.listdir(PATH +"/"+ dataset)

    for img in img_list:

        input_img = cv2.imread(PATH +"/"+dataset+"/"+img)

        input_img_resize=cv2.resize(input_img,(img_rows,img_cols))

        img_data_list.append(input_img_resize)

        target_column.append(dataset)
num_classes = len(classes_names_list)

print("num_classes",num_classes)

img_data = np.array(img_data_list) # convert images in numpy array 

img_data = img_data.astype('float32')

img_data /= 255

print("Shape of image data",img_data.shape)

num_of_samples = img_data.shape[0]

input_shape = img_data[0].shape 

print("number of samples",num_of_samples)

print("target column before encoding",target_column)
Labelencoder = LabelEncoder()

target_column = Labelencoder.fit_transform(target_column)

np.unique(target_column)
target_column
target_column_hotcoded = to_categorical(target_column,num_classes)

X,Y = shuffle(img_data,target_column_hotcoded,random_state=2)

X_train,X_temp,y_train,y_temp = train_test_split(X,Y,test_size=0.3,random_state=2)

X_test,X_val,y_test,y_val = train_test_split(X_temp,y_temp,test_size=0.3,random_state=2)
first_Mod = Sequential()



first_Mod.add(Conv2D(64,(3,3),activation='relu',input_shape=input_shape))

first_Mod.add(Conv2D(64,(3,3),activation='relu'))

first_Mod.add(MaxPool2D(pool_size=(2,2)))

first_Mod.add(Dropout(0.2))



first_Mod.add(Conv2D(128,(3,3),activation='relu'))

first_Mod.add(Conv2D(128,(3,3),activation='relu'))

first_Mod.add(MaxPool2D(pool_size=(2,2)))

first_Mod.add(Dropout(0.2))



first_Mod.add(Flatten())

first_Mod.add(Dense(128,activation='relu'))

first_Mod.add(Dropout(0.2))

first_Mod.add(Dense(num_classes,activation='softmax'))

first_Mod.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



first_Mod.summary()
hist = first_Mod.fit(X_train,y_train,batch_size=batch_size,epochs=num_epoch,verbose=1,validation_data=(X_test,y_test))

score = first_Mod.evaluate(X_test,y_test,batch_size=batch_size)

print('Test Loss',score[0])

print("Test Accuracy",score[1])