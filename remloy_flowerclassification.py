#import nessarary libraries

import tensorflow as tf

from tensorflow import keras

from keras.models import load_model



import matplotlib.pyplot as plt

import numpy as np

import os

import cv2

import pandas

import os

import random

from tqdm import tqdm
path = '../input/flowers/flowers'

folders = os.listdir(path)

print(folders)
image_names =[]

train_lables =[]

train_images =[]
# resizing and enumrating lists and transform it into numpy array



size = 64,64



for folder in folders:

  print(folder)

  for file in os.listdir(os.path.join(path,folder)):

    if file.endswith('jpg'):

      image_names.append(os.path.join(path,folder,file))

      train_lables.append(folder)

      img = cv2.imread(os.path.join(path,folder,file))

      img = cv2.resize(img,size)

      train_images.append(img)

    else:

      continue

      

train = np.array(train_images)

print(train.size)
train = train.astype('float32') / 255.0



# extract lables

label_dummies = pandas.get_dummies(train_lables)

labels =  label_dummies.values.argmax(1)
print(pandas.unique(train_lables))

print(pandas.unique(labels))
# Shuffle the labels and images randomly for better results



union_list = list(zip(train, labels))

random.shuffle(union_list)

train,labels = zip(*union_list)
# Convert the shuffled list to numpy array type

train = np.array(train)

labels = np.array(labels)
# Building a model

model = keras.Sequential([

    keras.layers.Flatten(input_shape = (64,64,3)),

    keras.layers.Dense(128,activation = tf.nn.tanh),

    keras.layers.Dense(5,activation = tf.nn.softmax)

])
# model parameters

model.compile(optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy']

)
model.fit(train,labels,epochs = 100)