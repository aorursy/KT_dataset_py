# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
print(os.listdir("../input/facesdata/FacesData"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras,os
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint

import cv2
from keras.layers.normalization import BatchNormalization
data_path ="../input/facesdata/FacesData/"

for dir1 in os.listdir(data_path):
    count=0
    for f in os.listdir(data_path + dir1):
        count+=1
    print(f"{dir1} has {count} images")
import random
folder_path="../input/facesdata/FacesData/aggression/"
a=random.choice(os.listdir(folder_path))
plt.imread(folder_path+a).shape
folder_path="../input/facesdata/FacesData/neutral/"
a=random.choice(os.listdir(folder_path))
plt.imread(folder_path+a).shape
folder_aggression  = '../input/facesdata/FacesData/aggression'
folder_neutral = '../input/facesdata/FacesData/neutral'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# Load in pictures
ims_aggression = [read(os.path.join(folder_aggression, filename)) for filename in os.listdir(folder_aggression)]
X_aggression = np.array(ims_aggression, dtype='uint8')
ims_neutral = [read(os.path.join(folder_neutral, filename)) for filename in os.listdir(folder_neutral)]
X_neutral = np.array(ims_neutral, dtype='uint8')
X_aggression.shape
X_neutral.shape   #need to reize it
plt.imshow(X_aggression[2]) 
plt.show()  # display it
X_neutral_1=[]
for img in range (len(X_neutral)):
    input_img_resize=cv2.resize(X_neutral[img],(48,48))
    X_neutral_1.append(input_img_resize)
X_neutral_1=np.array(X_neutral_1)
X_neutral_1.shape   #now we have the same size for each class with tensor
plt.imshow(X_neutral_1[2]) 
plt.show()  # display it
# Create labels
y_neutral = np.zeros(X_neutral_1.shape[0])
y_aggression = np.ones(X_aggression.shape[0])
X = np.concatenate((X_neutral_1, X_aggression), axis = 0)

# Merge data and shuffle it
X = np.concatenate((X_neutral_1, X_aggression), axis = 0)
y = np.concatenate((y_neutral, y_aggression), axis = 0)
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
y = y[s]
X.shape,y.shape
X_scaled = X/255

from keras.applications import VGG16

IMAGE_SIZE = [48, 48] 
# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (48,48,3) as required by VGG


print('Output_layer_type= {}'.format(vgg16_model.layers[-1]))
print('Output_layer_shape= {}'.format(vgg16_model.layers[-1].output_shape))
for layer in pretrained_model.layers:
    layer.trainable = False
x = Flatten()(vgg.output)

num_classes=1
x = Dense(num_classes, activation = 'sigmoid')(x)  # adding the output layer
model = Model(inputs = vgg.input, outputs = x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

from keras.preprocessing.image import ImageDataGenerator

trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
history = model.fit_generator(trainAug.flow(X_scaled, y),epochs=10)
