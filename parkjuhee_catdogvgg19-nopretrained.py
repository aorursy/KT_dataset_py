#### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import random

import os,shutil



src_path="../input"



print(os.listdir(src_path))



#constant value

VALID_SPIT=0.2

IMAGE_SIZE=80

BATCH_SIZE=20
label=[]

data=[]

counter=0

path="../input/train/train"

for file in os.listdir(path):

    image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_COLOR)

    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))

    if file.startswith("cat"):

        label.append(0)

    elif file.startswith("dog"):

        label.append(1)

    try:

        data.append(image_data/255)

    except:

        label=label[:len(label)-1]

    counter+=1

    if counter%1000==0:

        print (counter," image data retreived")



data=np.array(data)

data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],3)

label=np.array(label)

print (data.shape)

print (label.shape)
from sklearn.model_selection import train_test_split

train_data, valid_data, train_label, valid_label = train_test_split(

    data, label, test_size=0.2, random_state=42)

print(train_data.shape)

print(train_label.shape)

print(valid_data.shape)

print(valid_label.shape)
from keras import applications

vgg_model = applications.VGG16(weights=None,

                               include_top=False,

                               input_shape=(80, 80, 3))



vgg_model.summary()
from keras import layers

from keras import models

model = models.Sequential()

model.add(vgg_model)

model.add(layers.Flatten())

model.add(layers.Dense(256, activation  = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()
from keras import backend as K

K.set_image_dim_ordering('th')

K.set_image_data_format('channels_last')

from keras import layers

from keras import models

from keras import optimizers

from keras.layers import GlobalAveragePooling2D



model.compile(loss='binary_crossentropy',optimizer=optimizers.adam(lr=1e-4),metrics=['acc'])

# vgg_model.summary()

# train_history=vgg_model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=40,batch_size=BATCH_SIZE)
train_history=model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=20,batch_size=BATCH_SIZE)
import matplotlib.pyplot as plt

acc = train_history.history['acc']

val_acc = train_history.history['val_acc']

loss = train_history.history['loss']

val_loss = train_history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'blue', label='Training acc')

plt.plot(epochs, val_acc, 'red', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'blue', label='Training loss')

plt.plot(epochs, val_loss, 'red', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
from keras import Sequential

from keras.layers import *

import keras.optimizers as optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import *

import keras.backend as K

test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory("../input/test1",target_size=(80, 80),batch_size=32,class_mode='binary')
from tensorflow.python.keras.models import Sequential

from keras.models import load_model



print("-- Evaluate --")



scores = model.evaluate_generator(

            test_generator, 

            steps = 100)



print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
Y_pred = model.predict(valid_data)

predicted_label=np.round(Y_pred,decimals=2)
import matplotlib.pyplot as plt

%matplotlib inline

from mlxtend.plotting import plot_confusion_matrix



# Get the confusion matrix



CM = confusion_matrix(valid_label, Y_pred.round())

fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(12, 12))

plt.xticks(range(2), ['Cat', 'Dog'], fontsize=16)

plt.yticks(range(2), ['Cat', 'Dog'], fontsize=16)

plt.show()

test_data=[]

id=[]

counter=0

for file in os.listdir("../input/test1/test1"):

    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_COLOR)

    try:

        image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))

        test_data.append(image_data/255)

        id.append((file.split("."))[0])

    except:

        print ("ek gaya")

    counter+=1







test_data=np.array(test_data)

print (test_data.shape)

test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],3)

dataframe_output=pd.DataFrame({"id":id})
predicted_labels=model.predict(test_data)

predicted_labels=np.round(predicted_labels,decimals=2)

labels=[1 if value>0.5 else 0 for value in predicted_labels]
dataframe_output["label"]=labels

print(dataframe_output)
#0이 고양이 1이 강아지

import matplotlib.pyplot as plt

%matplotlib inline



for i in range(40):

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 2)

    plt.imshow(test_data[i])

#     plt.xlabel('label')

    plt.xlabel(dataframe_output["label"][i],fontsize=30)

    plt.title('0-cat,1-dog',fontsize=40)

    plt.show()

correct_indices = []

incorrect_indices = []



answer=[1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,0,0,0,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0]
correct_indices=[]

incorrect_indices=[]



for i in range(len(answer)):

  if dataframe_output["label"][i] == answer[i]:

    correct_indices.append(test_data[i])

  else:

    incorrect_indices.append(test_data[i])
print(len(correct_indices),len(incorrect_indices))
for i in range(len(correct_indices)):

    plt.figure(figsize=(20, 6))

    plt.subplot(1,2,1)

    plt.title('correct',fontsize=40)

    plt.imshow(correct_indices[i])

    plt.show()
for i in range(len(incorrect_indices)):

    plt.figure(figsize=(20, 6))

    plt.subplot(1,2,1)

    plt.title('incorrect',fontsize=40)

    plt.imshow(incorrect_indices[i])

    plt.show()