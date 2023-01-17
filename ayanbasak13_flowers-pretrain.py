# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data="../input/flowers-recognition/flowers/flowers/"
folders=os.listdir(data)
print(folders)
resnet_weights_path = '../input/resnet50-wts/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#print(os.listdir('../input/resnet50-wts/'))
#print(is_pathname_valid(resnet_weights_path))
# Any results you write to the current directory are saved as output.
img_names = []
train_labels = []
train_images = []

size = 64,64

for folder in folders :
    for file in os.listdir(os.path.join(data,folder)) :
        if(file.endswith('jpg')) :
            img_names.append(folder)
            train_labels.append(folder)
            img=cv2.imread(os.path.join(data,folder,file))
            im=cv2.resize(img,size)
            train_images.append(im)
        else :
            continue
        
train = np.array(train_images)
train.shape
# Reduce the RGB values between 0 and 1

train = train.astype('float32') / 255.0
label_dummies = pd.get_dummies(train_labels)
labels =  label_dummies.values.argmax(1)
print(labels)

pd.unique(train_labels)
pd.unique(labels)
# Shuffle the labels and images randomly for better results
import random
num_classes=5
union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)
train = np.array(train)
labels = np.array(labels)
out_y = keras.utils.to_categorical(labels, num_classes)
print(out_y)
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 5
resnet_weights_path = '../input/resnet50-wts/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

my_new_model.fit(train,out_y, epochs=2)

my_new_model.fit(train,out_y, epochs=1)
