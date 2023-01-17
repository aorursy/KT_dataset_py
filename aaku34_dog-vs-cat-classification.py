# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import cv2#open cv library for image reading

import matplotlib.pyplot as plt#for plotting of graph and visualising it 

%matplotlib inline
import os 

import random

import gc
train_dir='../input/train'

test_dir='../input/test'
train_dogs=['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]#get dog image

train_cats=['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]#get dog image
test_image=['../input/test/{}'.format(i) for i in os.listdir(test_dir) ]#get test image
train_image=train_dogs[:2000]+train_cats[:2000]#slice dataset upto 2000 samples

random.shuffle(train_image)
del train_dogs

del train_cats
gc.collect()#collect garbage to save memory
import matplotlib.image as mpimg#importing image into numpy array

for ima in train_image[0:3]:

    img=mpimg.imread(ima)

    imgplot = plt.imshow(img)#plotting image as numpy array

    plt.show()

    
nrows=150

ncolums=150

channel=3
#a function to read and process our data in acceptable format

def read_process(list_of_img):

    x=[]#features

    y=[]#labels

    for image in list_of_img:

        x.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncolums),interpolation=cv2.INTER_CUBIC))#read image

        #get labels

        if 'dog' in image:

            y.append(1)

        elif 'cat' in image:

            y.append(0)

    return x,y

x, y = read_process(train_image)

#x is now an array of image pixel values and y is a list of labels
#display 5 train image 

plt.figure(figsize=(20,10))

colums=5

for i in range(colums):

    plt.subplot(5/colums+1,colums,i+1)

    plt.imshow(x[i])
import seaborn as sns

del train_image

gc.collect()
#convert to array 

x=np.array(x)

y=np.array(y)
sns.countplot(y)

plt.title('labels for cats and dogs')
print(x.shape)

print(y.shape)
#spilit into train and validation set

from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=.20,random_state=2)

print(x_train.shape)

print(x_val.shape)

del x

del y

gc.collect()
n_train=len(x_train)

n_val=len(x_val)
batch_size=32
from keras.applications import InceptionResNetV2

conv_base=InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(150,150,3))
from keras import  layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array,load_img
#creating last layer and adding to to pre-trained model

model=models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
#freeze conv_base and train our own only

print("no of trainable weight before freezing:",len(model.trainable_weights))
conv_base.trainable=False

print("no of trainable weight after freezing:",len(model.trainable_weights))

#specify loss and optimizers

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=.0002),metrics=['acc'])
#data argumentation

train_datagen=ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)#only rescaling in validation set
#creating image generators

train_generator=train_datagen.flow(x_train,y_train,batch_size=batch_size)

val_generator=val_datagen.flow(x_val,y_val,batch_size=batch_size)
#training the dataset

history=model.fit_generator(train_generator,steps_per_epoch=n_train//batch_size,epochs=20,validation_data=val_generator,validation_steps=n_val//batch_size)
#saving trained models so tey can be reused

model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')
#predicting first 10 images

x_test,y_test=read_process(test_image[0:10])

X=np.array(x_test)

test_datagen=ImageDataGenerator(rescale=1./255)
#loop that iterates over the Images from the generator to make predictions. 

i=0

text_labels=[]

plt.figure(figsize=(30,20))

for batch in test_datagen.flow(X,batch_size=1):

    pred=model.predict(batch)

    if pred>0.5:

        text_labels.append('dogs')

    else:

         text_labels.append('cats')

    plt.subplot(5/colums+1,colums,i+1)

    plt.title(text_labels[i])

    imgplot=plt.imshow(batch[0])

    i+=1

    if i%10==0:

        break

    plt.show()

        
