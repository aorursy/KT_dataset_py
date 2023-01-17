# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import xml.etree.ElementTree as et

import matplotlib.pyplot as plt

import tensorflow as tf

train_image = []

train_label=[]

shape = (250,250)



for i in os.listdir("/kaggle/input/fruit-images-for-object-detection/train_zip/train"):

    row = []

    path = "/kaggle/input/fruit-images-for-object-detection/train_zip/train/"+i

# Read Image .JPG

    if path[-4:] == '.jpg':

        img = cv2.imread(path)

        img = cv2.resize(img,shape)

        #img = tf.cast(img, tf.float32)

        train_label.append(i.split('_')[0])

        train_image.append(img)

        



# To Parse from .XML 

# import xml.etree.ElementTree as et

#        if path[-4:] == '.jpg': 

#        xml = et.parse("/kaggle/input/fruit-images-for-object-detection/train_zip/train/"+i)

#        root = xml.getroot()

#        imga = root[1].text



#Turning to NumPy array

#train_image = tf.cast(train_image, tf.float32)

train_image = np.array(train_image)

#.astype(np.float64)



#Label Names 

label_dict = {i:x for i,x in enumerate(set(train_label))}
#One Hot Encoding Labels 

train_label = pd.get_dummies(train_label).values
plt.figure()



_,line = plt.subplots(1,4) 

for i in range(4):

    line[i].imshow(train_image[i])

    print(train_label[i])

 
from keras.utils import to_categorical

from keras.layers import Dense

from keras.layers import Conv2D

from keras.layers import Flatten

from keras.layers import AveragePooling2D



from keras.models import Sequential



model = Sequential()

model.add(Conv2D(32, kernel_size=(2,2), activation='relu',input_shape=(250,250,3,)))

model.add(Conv2D(60, kernel_size=(2,2), activation='relu'))

model.add(AveragePooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(10, activation ='relu'))

model.add(Dense(4, activation ='relu'))

model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy']

             )
from sklearn.model_selection import train_test_split

X_train,X_valid, y_train, y_valid = train_test_split(train_image,train_label,test_size = 0.2)
y_train
H = model.fit(X_train,y_train, batch_size = 30, epochs = 18, validation_data = (X_valid,y_valid))
plt.plot(H.history['val_accuracy'])