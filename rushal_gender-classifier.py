# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow.keras as ka

import matplotlib.pyplot as plt

import matplotlib.image as image

import cv2 as cv

import tensorflow.keras as ka

from sklearn import preprocessing

from sklearn.utils import shuffle

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))

os.chdir("../input/dataset1")
os.getcwd()
train,test,valid = os.listdir("dataset1")
test,train,valid
women ,men= os.listdir('dataset1/{}'.format(valid))

men,women
#train Dataset

def read_image(image_path):

    img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)

    img = cv.resize(img,(128,128))

    #plt.imshow(img)

    return np.array(img)

    

def prepare_data(train):

    image_data = []

    image_label = []

    for i in os.listdir("dataset1/{}".format(train)):

        for image_file in os.listdir("dataset1/{}/{}".format(train,i)):

            #print(i)

            image_path = "dataset1/{}/{}/{}".format(train,i,image_file)

            if i == "woman":

                image_data.append(read_image(image_path))

                image_label.append(0)

            if i == "man":

                image_data.append(read_image(image_path))

                image_label.append(1)

    return np.array(image_data),np.array(image_label)

        
x_train,y_train  = prepare_data(train)

x_test,y_test  = prepare_data(test)

x_valid,y_valid  = prepare_data(valid)
x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_valid.shape,y_valid.shape
x_train = x_train.reshape(len(x_train),128*128)

x_test = x_test.reshape(len(x_test),128*128)

x_valid = x_valid.reshape(len(x_valid),128*128)
x_train = preprocessing.normalize(x_train)

x_test = preprocessing.normalize(x_test)

x_valid = preprocessing.normalize(x_valid)
x_train,y_train = shuffle(x_train,y_train)

x_test,y_test = shuffle(x_test,y_test)

x_valid,y_valid = shuffle(x_valid,y_valid)
y_train = ka.utils.to_categorical(y_train)

y_test = ka.utils.to_categorical(y_test)

y_valid = ka.utils.to_categorical(y_valid)
l0 = ka.layers.Dense(64,activation='relu')

l1 = ka.layers.Dropout(0.5)

l2 = ka.layers.Dense(128,activation='relu')

l3 = ka.layers.Dropout(0.5)

l4 = ka.layers.Dense(2,activation='softmax')


model = ka.Sequential([l0,l1,l2,l3,l4])
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=50,validation_data=(x_valid,y_valid))
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(["Loss","Val Loss"])

model.evaluate(x_test,y_test)
y_pred = model.predict_classes(x_test)

