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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics

import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.layers import Conv2D
labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels.head()
#x = cv2.imread("../input/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg")
#plt.imshow(x,cmap='gray')
imgs = []
for x in labels['id']:
    x = cv2.imread("../input/dog-breed-identification/train/"+x+".jpg")
    #x = cv2.imread("../input/train/"+x+".jpg",0)
    x = 0.2989 * x[:,:,0] + 0.5870 * x[:,:,1] + 0.1140 * x[:,:,2]
    x = cv2.resize(x, (64,64))
    #x = x/255.0
    x = (x-np.mean(x))/np.std(x)
    imgs.append(x)
len(imgs)
plt.imshow(imgs[0],cmap='gray')
#from collections import Counter
#x = dict(Counter([x.shape for x in imgs]).most_common(20))
#x
"""plt.figure(figsize=(12,8))
plt.bar(range(20),x.values())
plt.xticks(range(20),x.keys(), rotation='vertical')
plt.show()"""
"""x = Counter([x for x in labels['breed']])
plt.figure(figsize=(20,8))
plt.bar(range(len(x)),x.values())
plt.xticks(range(len(x)),x.keys(), rotation='vertical')
plt.show()"""
"""def preproc(image):
    x = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
    x = cv2.resize(x, (64,64))
    x = (x-np.mean(x))/np.std(x)
    return x"""
#prepro_imgs = [preproc(img) for img in imgs]
#X = np.array(prepro_imgs)
X = np.array(imgs)
Y = pd.get_dummies(labels['breed'])
data_aug =ImageDataGenerator(width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             rotation_range = 40,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             fill_mode = 'nearest')
"""image = cv2.imread("../input/train/000bec180eb18c7604dcecc8fe0dba07.jpg")
pro_image = preproc(image)
plt.imshow(pro_image,cmap='gray')"""
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 10)
print (x_train.shape)
print (y_train.shape)
X_train = x_train[:,:,:,np.newaxis]
X_test = x_test [:,:,:,np.newaxis]
data_aug.fit(X_train)
train_gen = data_aug.flow(np.array(X_train),np.array(y_train),batch_size=32)
test_gen = data_aug.flow(np.array(X_test),np.array(y_test),batch_size=32)
from keras.layers import MaxPooling2D,ZeroPadding2D,BatchNormalization
model = Sequential()
model.add(Conv2D(32,input_shape=(64,64,1),kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()
model.compile(optimizer='Adam',
          loss='categorical_crossentropy', 
           metrics=['accuracy'])
model.fit_generator(train_gen,steps_per_epoch=len(x_train)/32, epochs=30, validation_data = test_gen,validation_steps = len(X_test)/32)
tested = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
print (tested.head())
test_imgs = []
for x in tested['id']:
    x = cv2.imread("../input/dog-breed-identification/test/"+x+".jpg")
    #x = cv2.imread("../input/train/"+x+".jpg",0)
    x = 0.2989 * x[:,:,0] + 0.5870 * x[:,:,1] + 0.1140 * x[:,:,2]
    x = cv2.resize(x, (64,64))
    #x = x/255.0
    x = (x-np.mean(x))/np.std(x)
    test_imgs.append(x)
len(test_imgs)
test = np.array(test_imgs)[:,:,:,np.newaxis]
output= model.predict(test)
output
output[0].argmax()
output[0].max()
sub = pd.DataFrame(output,columns=pd.get_dummies(labels['breed']).columns)
sub.insert(loc=0, column='id', value=tested['id'])
sub.drop(0)
sub.head()
sub.to_csv('ouput.csv',index=False)
