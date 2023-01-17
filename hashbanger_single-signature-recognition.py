# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import sys

import csv

import os

import math

import json, codecs

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

# from zipfile import ZipFile

# import shutil

import glob

from PIL import Image

from PIL import ImageFilter

from sklearn.model_selection import train_test_split

import keras

from keras import backend as K

from keras import layers

from keras.models import Model

from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.models import Sequential, load_model

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications import ResNet50

from keras.applications.resnet50 import preprocess_input

# import torch 

# import torch.nn as nn

# import torch.nn.functional as F

import time

import cv2

import h5py
import re

numbers = re.compile(r'(\d+)')

def numericalSort(value):

    parts = numbers.split(value)

    parts[1::2] = map(int, parts[1::2])

    return parts
genuine_paths = sorted(glob.glob('../input/signature-single-sample/signatures_single/test/genuine/*'), key=numericalSort)\

+ sorted(glob.glob('../input/signature-single-sample/signatures_single/train/genuine/*'), key=numericalSort)

# train_f = sorted(glob.glob('signature22/train/forged/*'), key=numericalSort)



# for i in train_f:

#     X_train_paths.append(i)

    

forged_paths = sorted(glob.glob('../input/signature-single-sample/signatures_single/test/forged/*'), key=numericalSort)\

+ sorted(glob.glob('../input/signature-single-sample/signatures_single/train/forged/*'), key=numericalSort)

# test_f = sorted(glob.glob('signature22/test/forged/*'), key=numericalSort)



# for i in test_f:

#     X_test_paths.append(i)
genuine_eq = []

genuine_eq = [cv2.imread(path) for path in genuine_paths]



forged_eq = []

forged_eq = [cv2.imread(path) for path in forged_paths]
plt.imshow(genuine_eq[2]);
plt.imshow(forged_eq[0]);
# dst = cv2.fastNlMeansDenoisingColored(genuine_eq[0], None, 10, 10, 7, 15)

# plt.figure(figsize=(18,4))

# plt.subplot(1,2,1)

# plt.imshow(genuine_eq[0])

# plt.subplot(1,2,2)

# plt.imshow(dst);
# for i in range(len(genuine_eq)):

#     genuine_eq[i] = cv2.fastNlMeansDenoisingColored(genuine_eq[i], None, 10, 10, 7, 15)



# for i in range(len(forged_eq)):

#     forged_eq[i] = cv2.fastNlMeansDenoisingColored(forged_eq[i], None, 10, 10, 7, 15)
for image in genuine_eq:

    image[np.where((image > [0,0,200]).all(axis=2))] = [255,255,255]



for image in forged_eq:

    image[np.where((image > [0,0,200]).all(axis=2))] = [255,255,255]
plt.figure(figsize=(20,9))

plt.subplot(2,4,1)

plt.imshow(genuine_eq[0])

plt.subplot(2,4,2)

plt.imshow(genuine_eq[1])

plt.subplot(2,4,3)

plt.imshow(genuine_eq[2])

plt.subplot(2,4,4)

plt.imshow(genuine_eq[3])

plt.subplot(2,4,5)

plt.imshow(forged_eq[4])

plt.subplot(2,4,6)

plt.imshow(forged_eq[5])

plt.subplot(2,4,7)

plt.imshow(forged_eq[6])

plt.subplot(2,4,8)

plt.imshow(forged_eq[7])

plt.show()
os.makedirs('../kaggle/working/processed/train/genuine/')

os.makedirs('../kaggle/working/processed/test/genuine/')

os.makedirs('../kaggle/working/processed/train/forged/')

os.makedirs('../kaggle/working/processed/test/forged/')
# for i in range(len(genuine_eq[:20])):

#     cv2.imwrite('kaggle/working/processed/train/genuine/'+str(i)+'.png',genuine_eq[i])

# for i in range(len(genuine_eq[20:24])):

#     cv2.imwrite('kaggle/working/processed/test/genuine/'+str(i)+'.png',genuine_eq[i])

    

# for i in range(len(forged_eq[:20])):

#     cv2.imwrite('kaggle/working/processed/train/forged/'+str(i)+'.png',forged_eq[i])

# for i in range(len(forged_eq[20:24])):

#     cv2.imwrite('kaggle/working/processed/test/forged/'+str(i)+'.png',forged_eq[i])



# for i in range(len(genuine_eq[0])):

cv2.imwrite('../kaggle/working/processed/train/genuine/'+str(23)+'.png',genuine_eq[23])

for i in range(len(genuine_eq[0:23])):

    cv2.imwrite('../kaggle/working/processed/test/genuine/'+str(i)+'.png',genuine_eq[i])

    

# for i in range(len(forged_eq[0])):

cv2.imwrite('../kaggle/working/processed/train/forged/'+str(23)+'.png',forged_eq[23])

for i in range(len(forged_eq[0:23])):

    cv2.imwrite('../kaggle/working/processed/test/forged/'+str(i)+'.png',forged_eq[i])

# for i in range(len(genuine_eq_inv)):

#     cv2.imwrite('processed/genuine/'+str(i)+'.png',genuine_eq_inv[i])

    

# for i in range(len(forged_eq_inv)):

#     cv2.imwrite('processed/forged/'+str(i)+'.png',forged_eq_inv[i])

    

# for i in range(len(genuine_eq)//2):

#     cv2.imwrite('../input/output/processed/train/genuine/'+str(i)+'.png',genuine_eq[i])

#     cv2.imwrite('processed/test/genuine/'+str(i+4)+'.png',genuine_eq[i+4])

    

# for i in range(len(forged_eq)//2):

#     cv2.imwrite('processed/train/forged/'+str(i)+'.png',forged_eq[i])

#     cv2.imwrite('processed/test/forged/'+str(i+4)+'.png',forged_eq[i+4])    
train_path = "../kaggle/working/processed/train"

test_path = "../kaggle/working/processed/test"
test_g = []

test_f = []

for dirname, _, filenames in os.walk(test_path+'/genuine'):

    for filename in filenames:

        test_g.append(os.path.join(dirname, filename))



for dirname, _, filenames in os.walk(test_path+'/forged'):

    for filename in filenames:

        test_f.append(os.path.join(dirname, filename))



train_g = []

train_f = []

for dirname, _, filenames in os.walk(train_path+'/genuine'):

    for filename in filenames:

        train_g.append(os.path.join(dirname, filename))



for dirname, _, filenames in os.walk(train_path+'/forged'):

    for filename in filenames:

        train_f.append(os.path.join(dirname, filename))
plt.figure(figsize = (35,5))

plt.suptitle('Train Real Signatures', fontsize = 18)

x, y = 1, 1

for i in range(1):

    plt.subplot(x, y, i+1)

    plt.axis('off')

    plt.imshow(cv2.resize(cv2.imread(train_g[i], 1), (224,224)))

plt.savefig('train_g')

plt.show()    
plt.figure(figsize = (35,5))

plt.suptitle('Train Fake Signatures', fontsize = 18)

x, y = 1, 1

for i in range(1):

    plt.subplot(x, y, i+1)

    plt.axis('off')

    plt.imshow(cv2.resize(cv2.imread(train_f[i], 1), (224,224)))

plt.savefig('train_f')

plt.show()
plt.figure(figsize = (35,5))

plt.suptitle('Test Real Signatures', fontsize = 18)

x, y = 1, 4

for i in range(4):

    plt.subplot(x, y, i+1)

    plt.axis('off')

    plt.imshow(cv2.resize(cv2.imread(test_g[i], 1), (224,224)))

plt.savefig('test_g')

plt.show()    
plt.figure(figsize = (35,5))

plt.suptitle('Test Fake Signatures', fontsize = 18)

x, y = 1, 4

for i in range(4):

    plt.subplot(x, y, i+1)

    plt.axis('off')

    plt.imshow(cv2.resize(cv2.imread(test_f[i], 1), (224,224)))

plt.savefig('test_f')

plt.show()    
numberOfClass = len(glob.glob(train_path+"/*"))

train_data = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), class_mode='binary')

test_data = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), class_mode='binary')
# Data replication



train_datagen = ImageDataGenerator(

    shear_range=10,

    zoom_range=0.2,

    horizontal_flip=True,

    preprocessing_function=preprocess_input)

 

train_generator = train_datagen.flow_from_directory(

    train_path,

    batch_size=32,

    class_mode='binary',

    target_size=(224,224))

 

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input)

 

test_generator = test_datagen.flow_from_directory(

    test_path,

    shuffle=False,

    class_mode='binary',

    target_size=(224,224))
vgg = VGG16()

vgg_layer_list = vgg.layers



model_vgg = Sequential()

for i in range(len(vgg_layer_list)-1):

    model_vgg.add(vgg_layer_list[i])



for layers in model_vgg.layers:

    layers.trainable = False    

model_vgg.add(Dense(numberOfClass, activation="softmax"))

print(model_vgg.summary())
model_vgg.compile(loss='sparse_categorical_crossentropy',

      optimizer='RMSProp',

      metrics=['accuracy'])



history = model_vgg.fit_generator(

    generator=train_generator,

    epochs=20,

#     validation_data=test_generator

)
# plt.figure(figsize = (35,5))

# plt.suptitle('Real signatures Train', fontsize=20)

# x, y = 1, 1

# for i in range(1):

#     plt.subplot(x, y, i+1)

#     plt.axis('off')

#     img = cv2.resize(cv2.imread(train_g[i], 1), (224,224))

#     plt.imshow(img)

#     img = img_to_array(img)

#     img = img.reshape(1,224,224,3)

#     pre = model_vgg.predict_classes(img, batch_size=1)

#     plt.title(pre, fontsize=20)

# plt.savefig('test_g_vgg')

# plt.show()





# plt.figure(figsize = (35,5))

# plt.suptitle('Fake signatures Train', fontsize=20)

# x, y = 1, 1

# for i in range(1):

#     plt.subplot(x, y, i+1)

#     plt.axis('off')

#     img = cv2.resize(cv2.imread(train_f[i], 1), (224,224))

#     plt.imshow(img)

#     img = img_to_array(img)

#     img = img.reshape(1,224,224,3)

#     pre = model_vgg.predict_classes(img, batch_size=1)

#     plt.title(pre, fontsize=20)

# plt.savefig('test_f_vgg')

# plt.show()
# plt.figure(figsize = (35,5))

# plt.suptitle('Real signatures Test', fontsize=20)

# x, y = 1, 23

# for i in range(23):

#     plt.subplot(x, y, i+1)

#     plt.axis('off')

#     img = cv2.resize(cv2.imread(test_g[i], 1), (224,224))

#     plt.imshow(img)

#     img = img_to_array(img)

#     img = img.reshape(1,224,224,3)

#     pre = model_vgg.predict_classes(img, batch_size=1)

#     plt.title(pre, fontsize=20)

# plt.savefig('test_g_vgg')

# plt.show()





# plt.figure(figsize = (35,5))

# plt.suptitle('Fake signatures Test', fontsize=20)

# x, y = 1, 23

# for i in range(23):

#     plt.subplot(x, y, i+1)

#     plt.axis('off')

#     img = cv2.resize(cv2.imread(test_f[i], 1), (224,224))

#     plt.imshow(img)

#     img = img_to_array(img)

#     img = img.reshape(1,224,224,3)

#     pre = model_vgg.predict_classes(img, batch_size=1)

#     plt.title(pre, fontsize=20)

# plt.savefig('test_f_vgg')

# plt.show()
genuine_pred = []

for i in range(23):

    img = cv2.resize(cv2.imread(test_g[i], 1), (224,224))

    img = img_to_array(img)

    img = img.reshape(1,224,224,3)

    genuine_pred.append(model_vgg.predict_classes(img, batch_size=1))

    

forged_pred = []

for i in range(23):

    img = cv2.resize(cv2.imread(test_f[i], 1), (224,224))

    img = img_to_array(img)

    img = img.reshape(1,224,224,3)

    forged_pred.append(model_vgg.predict_classes(img, batch_size=1))
(np.array(genuine_pred).flatten().sum() / len(genuine_pred)) * 100
abs(np.array(forged_pred).flatten() - 1).sum() / len(forged_pred) * 100