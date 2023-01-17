#This Python 3 environment comes with many helpful analytics libraries installed
#It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
#For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Input data files are available in the "../input/" directory.
#For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Any results you write to the current directory are saved as output.
import tensorflow as tsf
import keras
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
y= train.iloc[:,1:].values
print(y)
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
import cv2
img_size=150
train_image = []
for name in train['image_id']:
    path='../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'
    img=cv2.imread(path)
    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
    train_image.append(image)
test_image = []

for name in test['image_id']:
    path='../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'
    img=cv2.imread(path)
    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)
    test_image.append(image)
x_train = np.asarray(train_image, dtype=np.float32)
x_train = x_train/255



x_test = np.asarray(test_image, dtype=np.float32)
x_test = x_test/255

x_test.shape
y_train = np.array(y, dtype='float32')
from sklearn.model_selection import train_test_split
# resplit and shape the data again. the data to get a clean set.
x_train, x_val, y_train, y_val = train_test_split(x_train, 
                                                  y_train, 
                                                  test_size = 0.2, 
                                                  random_state = 2000 )
print(x_train[1:],x_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
#create model
model = Sequential()

#add model layers
model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape= x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=4, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
print(y_train.shape,y_val.shape)
model.fit(x_train, y_train,validation_data=(x_val,y_val), epochs=20,batch_size=32)
y_pred=model.predict(x_test)
print(y_pred)
testset = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
Id= testset.iloc[:,:].values
import csv
with open(r'/kaggle/working/Plant_pathology.csv',mode='w') as plants:
    writer= csv.writer(plants,delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['image_id','healthy','multiple_diseases','rust','scab'])
    for i in range(len(Id)):
        writer.writerow([Id[i][0],y_pred[i][0],y_pred[i][1],y_pred[i][2],y_pred[i][3]])