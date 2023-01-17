# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!unzip ../input/facial-keypoints-detection/training.zip -d train
!unzip ../input/facial-keypoints-detection/test.zip -d test
train = pd.read_csv("../working/train/training.csv")
test = pd.read_csv("../working/test/test.csv")
test.shape
print(test)
train.columns
train.head().T
train.fillna(method='ffill',inplace=True)
print(train)
len(train["Image"][4].split(' '))
images = np.ndarray((7049,9216))
for i in range(7049):
    img = np.array(train["Image"][i].split(' '))
    img = ['0' if x == '' else x for x in img]
    images[i,:] = img
Y_test = np.ndarray((1783,9216))
for i in range(1783):
    img = np.array(test["Image"][i].split(' '))
    img = ['0' if x == '' else x for x in img]
    Y_test[i,:] = img
images = images.reshape(-1,96,96,1)
Y_test = Y_test.reshape(-1,96,96,1)
images.shape
plt.imshow(images[34].reshape(96,96),cmap='gray')
train.columns
train.drop('Image',axis=1)
Y_train = np.array(train.drop("Image",axis=1),dtype='float')
print(Y_train.shape)
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(96,96,1),padding = 'SAME',activation='relu'))
model.add(Conv2D(32,(3,3),padding = 'SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding = 'SAME',activation='relu'))
model.add(Conv2D(64,(3,3),padding = 'SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.4))
          
model.add(Conv2D(128,(3,3),padding = 'SAME',activation='relu'))
model.add(Conv2D(128,(3,3),padding = 'SAME',activation='relu'))
model.add(Conv2D(128,(3,3),padding = 'SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.4))
          
model.add(Conv2D(256,(3,3),padding = 'SAME',activation='relu'))
model.add(Conv2D(256,(3,3),padding = 'SAME',activation='relu'))
model.add(Conv2D(256,(3,3),padding = 'SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512,(3,3),padding = 'SAME',activation='relu'))
model.add(Conv2D(512,(3,3),padding = 'SAME',activation='relu'))
model.add(Conv2D(512,(3,3),padding = 'SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.4))
          
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=30))
          
model.summary()
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])
model.fit(images,Y_train,epochs=30,batch_size=256,validation_split=0.2)
pred = model.predict(Y_test)
lookid_data = pd.read_csv("/kaggle/input/facial-keypoints-detection/IdLookupTable.csv")
lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(pred)
rowid = lookid_data['RowId']
rowid=list(rowid)
feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))
preded = []
for x,y in zip(imageID,feature):
    preded.append(pre_list[x][y])
rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)
submission.to_csv('face_key_detection_submission.csv',index = False)