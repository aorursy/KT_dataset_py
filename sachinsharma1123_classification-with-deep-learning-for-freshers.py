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
from glob import glob

data = glob('/kaggle/input/breast-histopathology-images/**/*.png', recursive=True)
import cv2

import matplotlib.pyplot as plt

for i in data[:5]:

    img=cv2.imread(i)

    img_1=cv2.resize(img,(200,200))

    plt.imshow(img_1,cmap='binary')

    plt.show()
images=[]

labels=[]

for i in data[:15000]:

    if i.endswith('.png'):

        label=i[-5]

        img=cv2.imread(i)

        img_1=cv2.resize(img,(100,100))

        images.append(img_1)

        labels.append(label)
x=np.stack(images)
from tensorflow.keras.utils import to_categorical

y=to_categorical(labels)
#normalize the data

x=x/255
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Conv2D,MaxPool2D,Flatten

from tensorflow.keras.models import Sequential

from tensorflow.keras import regularizers

model=Sequential([

    Conv2D(64,(3,3),activation='relu',input_shape=(100,100,3)),

    MaxPool2D((2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(128,(3,3),activation='relu',padding='same'),

    MaxPool2D((2,2)),

    BatchNormalization(),

    Dropout(0.25),

    Conv2D(256,(3,3),activation='relu',padding='same'),

    MaxPool2D((2,2)),

    BatchNormalization(),

    Dropout(0.3),

    Conv2D(512,(3,3),activation='relu',padding='same'),

    MaxPool2D((2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Flatten(),

    Dense(1024,activation='relu'),

    Dense(2,activation='sigmoid')

])
model.compile(optimizer='Adam',loss='mae',metrics=['acc'])
history=model.fit(x,y,epochs=15,validation_split=0.3,batch_size=56)
loss,accuracy=model.evaluate(x_test,y_test)
plt.figure(figsize=(12,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(['train_data','test_data'])

plt.title('loss analysis')

plt.show()
plt.figure(figsize=(12,5))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend(['train_data','test_data'])

plt.title('accuracy analysis')

plt.show()
