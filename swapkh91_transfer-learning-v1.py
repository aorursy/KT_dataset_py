# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from keras.applications import VGG19, ResNet50

import cv2

import os

import random

import tensorflow as tf

import gc
labels = ['dandelion', 'daisy','tulip','sunflower','rose']

img_size = 224

def get_data(data_dir):

    data = [] 

    for label in labels: 

        path = os.path.join(data_dir, label)

        class_num = labels.index(label)

        for img in os.listdir(path):

            try:

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size

                data.append([resized_arr, class_num])

            except Exception as e:

                print(e)

    return np.array(data)
data = get_data("/kaggle/input/flowers-recognition/flowers")
# visualize some random images

fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=random.randint(0,len(data))

        ax[i,j].imshow(data[l][0])

        ax[i,j].set_title('Flower: '+labels[data[l][1]])

        

plt.tight_layout()
# separate label and features

x = []

y = []



for feature, label in data:

    x.append(feature)

    y.append(label)
# Normalize the data

x = np.array(x) / 255
#reshape to 3D image

x = x.reshape(-1, img_size, img_size, 3)

y = np.array(y)
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

y = label_binarizer.fit_transform(y)
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)
# clear RAM

del(x)

del(y)

del(data)
gc.collect()
# resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
# load VGG19 model with weights and exclude final layer as it has to be according to our problem statement

# i.e. 5 classes to predict

vgg = VGG19(input_shape=(224,224,3), include_top=False, weights="imagenet")
# for i, layer in enumerate(resnet50.layers):

#     print(i, layer.name, layer.trainable)



for i, layer in enumerate(vgg.layers):

    print(i, layer.name, layer.trainable)
# for layer in resnet50.layers:

#     layer.trainable = True



# for layer in resnet50.layers[:143]:

#     layer.trainable = False

    

# for layer in vgg.layers:

#     layer.trainable = True



# do not train the first 18 layers

for layer in vgg.layers[:19]:

    layer.trainable = False
# create a new model according to problem statement using pre-trained VGG model and some more layers

model=Sequential()

# model.add(resnet50)

model.add(vgg)

model.add(MaxPool2D((2,2) , strides = 2))

model.add(Flatten())

# model.add(Dense(256,activation='relu'))

model.add(Dense(5,activation='softmax'))
model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
# optimize LR

from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.6, min_lr=0.000001)
history = model.fit(x_train,y_train, batch_size = 64 , epochs = 50 , 

                    validation_data = (x_test, y_test),callbacks = [learning_rate_reduction])
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
del(model)

del(history)
gc.collect()
# now trying with freezing first 19 layers

for layer in vgg.layers:

    layer.trainable = True



for layer in vgg.layers[:20]:

    layer.trainable = False
model=Sequential()

# model.add(resnet50)

model.add(vgg)

model.add(MaxPool2D((2,2) , strides = 2))

model.add(Flatten())

# model.add(Dense(256,activation='relu'))

model.add(Dense(5,activation='softmax'))



model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
history = model.fit(x_train,y_train, batch_size = 64 , epochs = 25, 

                    validation_data = (x_test, y_test),callbacks = [learning_rate_reduction])
del(model)

del(history)
gc.collect()
# freezing 19 layers didn't give improvement over previous model so moving back to 18

for layer in vgg.layers:

    layer.trainable = True



for layer in vgg.layers[:19]:

    layer.trainable = False
# adding a Dense layer to see if improves the model

model=Sequential()

# model.add(resnet50)

model.add(vgg)

# model.add(MaxPool2D((2,2) , strides = 2))

# model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Flatten())

model.add(Dense(5,activation='softmax'))



model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
history = model.fit(x_train,y_train, batch_size = 64 , epochs = 25, 

                    validation_data = (x_test, y_test),callbacks = [learning_rate_reduction])
# this shows the model has a slight overfitting

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
predictions = model.predict_classes(x_test)

y_test_inv = label_binarizer.inverse_transform(y_test)



i=0

prop_class=[]

mis_class=[]



for i in range(len(y_test_inv)):

    if(y_test_inv[i] == predictions[i]):

        prop_class.append(i)

    if(len(prop_class)==8):

        break



i=0

for i in range(len(y_test_inv)):

    if(y_test_inv[i] != predictions[i]):

        mis_class.append(i)

    if(len(mis_class)==8):

        break
# visualize some correctly predicted data

count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x_test[prop_class[count]])

        ax[i,j].set_title("Predicted Flower : "+ labels[predictions[prop_class[count]]] +"\n"+"Actual Flower : "+ labels[y_test_inv[prop_class[count]]])

        plt.tight_layout()

        count+=1
# visualize some incorrectly predicted data

count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x_test[mis_class[count]])

        ax[i,j].set_title("Predicted Flower : "+labels[predictions[mis_class[count]]]+"\n"+"Actual Flower : "+labels[y_test_inv[mis_class[count]]])

        plt.tight_layout()

        count+=1