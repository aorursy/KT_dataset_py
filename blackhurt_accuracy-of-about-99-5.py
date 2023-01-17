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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
import cv2
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
train_csv=pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Train.csv')
test_csv=pd.read_csv('/kaggle/input/gtsrb-german-traffic-sign/Test.csv')


train_X=train_csv['Path'].tolist()
test_X=test_csv['Path'].tolist()

train_y=np.array(train_csv['ClassId'])


train_X,train_y=shuffle(train_X,train_y)
trains=[]
for i in range(0,len(train_X)):
    source="/kaggle/input/gtsrb-german-traffic-sign/"+train_X[i]
    img=cv2.imread(source,1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(30,30))
    trains.append(img)
trains=np.array(trains)
trains=trains/255.0

  
    
trains_x,valids_x,trains_y,valids_y=train_test_split(trains,train_y,test_size=0.2,random_state=33)

trains_x.shape
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30,30,3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam', 
    metrics=['accuracy']
)
model.fit(trains_x, trains_y, epochs=30,verbose=2,validation_split=0.3)
