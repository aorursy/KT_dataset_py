# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
from sklearn.model_selection import train_test_split
TRAIN_DIR_PATH = '/kaggle/input/intel-image-classification/seg_train/seg_train/'
TEST_DIR_PATH = '/kaggle/input/intel-image-classification/seg_test/seg_test/'
PRED_DIR_PATH = '/kaggle/input/intel-image-classification/seg_pred/seg_pred/'
labels = ['buildings','glacier','sea','mountain','forest','street']
labels_transform = {'buildings':0, 'glacier':1, 'sea': 2, 'mountain': 3, 'forest':4, 'street': 5}
train_labels = []
test_labels = []

train_images = []
test_images = []
pred_images = []
for label in labels:
    train_img_path = os.path.join(TRAIN_DIR_PATH,label)
    train_img_file = os.listdir(train_img_path)
    
    for item in train_img_file:
        img = cv.imread(os.path.join(train_img_path,item),0)
        img = cv.resize(img,(150,150))
        train_images.append(img)
        train_labels.append(labels_transform[label])
        
    test_img_path = os.path.join(TEST_DIR_PATH,label)
    test_img_file = os.listdir(test_img_path)
    
    for item in test_img_file:
        img = cv.imread(os.path.join(test_img_path,item),0)
        img = cv.resize(img, (150,150))
        test_images.append(img)
        test_labels.append(labels_transform[label])
        
    
    
pred_img_file = os.listdir(PRED_DIR_PATH)
for item in pred_img_file:
    img = cv.imread(os.path.join(PRED_DIR_PATH,item),0)
    img = cv.resize(img,(150,150))
    pred_images.append(img)
train_dataset = np.zeros([len(train_images),1,150,150])
test_dataset = np.zeros([len(test_images),1,150,150])
pred_dataset = np.zeros([len(pred_images),1,150,150])


for i in range(len(train_images)):
    train_dataset[i] = train_images[i]

for i in range(len(test_images)):
    test_dataset[i] = test_images[i]

for i in range(len(pred_images)):
    pred_dataset[i] = pred_images[i]
    
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

print(train_dataset.shape)
print(test_dataset.shape)
print(train_labels.shape)
print(test_labels.shape)
print(pred_dataset.shape)
train_dataset = train_dataset/ 255.0
test_dataset = test_dataset/255.0
pred_dataset = pred_dataset / 255.0
model = keras.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',data_format = 'channels_first', padding='valid'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.4))
#model.add(keras.layers.Conv2D(64,(3,3),activation='relu',data_format = 'channels_first',padding='valid'))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(keras.layers.Dropout(0.4))
#model.add(keras.layers.Conv2D(64,(3,3),activation='relu',data_format = 'channels_first',padding='valid'))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(keras.layers.Dropout(0.4))
#model.add(keras.layers.Conv2D(64,(3,3),activation='relu',data_format = 'channels_first',padding='valid'))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
#model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500,activation='relu'))
#model.add(keras.layers.Dense(300,activation='relu'))
#model.add(keras.layers.Dense(100,activation='relu'))
#model.add(keras.layers.Dense(50,activation='relu'))
model.add(keras.layers.Dense(6,activation='softmax'))

model.compile(optimizer = 'Adam', loss='categorical_crossentropy',metrics=['accuracy'])

train_labels = keras.utils.to_categorical(train_labels,num_classes=None)
test_labels = keras.utils.to_categorical(test_labels,num_classes=None)
model.fit(train_dataset,train_labels,epochs=10)
model.fit(test_dataset,test_labels,epochs=10)

predictions = model.predict(pred_dataset)
plt.figure(figsize=(20,20))
for i in range(50):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pred_dataset[i][0])
    plt.xlabel(labels[np.argmax(predictions[i])])
plt.show()