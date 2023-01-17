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
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:23:18 2020

@author: Manish Sehrawat
"""

#HackerEarth
import os
import cv2
import keras
import tensorflow as tf
import numpy as np
import pandas as pd

TRAIN_DATA_DIREC = "/kaggle/input/dataset/Sample Data"

#reading the dataset
def get_dataset ():
    
    dataset = []
    for label_dir in os.listdir(TRAIN_DATA_DIREC):
        path = os.path.join(TRAIN_DATA_DIREC,label_dir)
        if not os.path.isdir(path):
            continue
        for image in os.listdir(path):
            image = cv2.imread(os.path.join(path,image))
            image = preprocess(image)
            dataset.append([image,label_dir])
    return zip(*dataset)

def preprocess(img):
    
    width = 300
    height = 300
    dimensions = (width,height)
    img = cv2.resize(img,dimensions,interpolation = cv2.INTER_LINEAR)
    return img

X,y = get_dataset()
Class_map = {'Adults':0,'Teenagers':1,'Toddlers':2}
Inverse_class_map = {0:'Adults',1:'Teenagers',2:'Toddlers'}
y = pd.Series(y)
y = y.map(Class_map)

# Model  Buildind :

from keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization

width = 300
height = 300
input_shape = (width,height,3)
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input
base_model=VGG19(include_top=False, weights='imagenet',input_shape=(width,height,3), pooling='max')

for layer in base_model.layers[:-4]:
    layer.trainable = False

for layer in base_model.layers:
    print(layer, layer.trainable)

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(8192, activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile( optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
X = np.array(X)
y = np.array(y)

model.fit(X,y,batch_size = 128 ,epochs = 100,verbose = 1)
test_img=[]
from tqdm import tqdm
path='/kaggle/input/testdata/Test Data'
test = pd.read_csv("/kaggle/input/test-csv/Test.csv")
for i in tqdm(test['Filename']):
    final_path=os.path.join(path,i)
    img=cv2.imread(final_path)
    img = preprocess(img)
    test_img.append(img)   
test_img = np.array(test_img)
labels = model.predict(test_img)
print(labels[:4])
label = [np.argmax(i) for i in labels]
class_label = [Inverse_class_map[x] for x in label]
print(class_label[:3])
submission = pd.DataFrame({ 'Filename': test.Filename, 'Category': class_label })
submission.head(10)
submission.to_csv('submission.csv', index=False)
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
base_model_2=VGG16(include_top=False, weights='imagenet',input_shape=(width,height,3), pooling='max')

for layer in base_model_2.layers[:]:
    layer.trainable = True
for layer in base_model_2.layers:
    print(layer, layer.trainable)


model2 = Sequential()
model2.add(base_model_2)
model2.add(Flatten())
model2.add(Dense(1024, activation='relu'))
model2.add(Dense(3,activation='softmax'))

model2.compile( optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model2.summary()
X = np.array(X)
y = np.array(y)
model2.fit(X,y,batch_size = 128 ,epochs = 80,verbose = 1)
test_img = np.array(test_img)
labels = model2.predict(test_img)
print(labels[:4])
label = [np.argmax(i) for i in labels]
class_label = [Inverse_class_map[x] for x in label]
print(class_label[:3])
submission = pd.DataFrame({ 'Filename': test.Filename, 'Category': class_label })
submission.head(10)
submission.to_csv('submission.csv', index=False)
