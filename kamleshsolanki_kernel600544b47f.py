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
import csv

import tensorflow as tf

import numpy as np

train_file = '/kaggle/input/sign-language-mnist/sign_mnist_train.csv'

test_file = '/kaggle/input/sign-language-mnist/sign_mnist_test.csv'
train_data = []

train_label = []



test_data = []

test_label = []

with open(train_file)  as csvfile:

    csvfile = csv.reader(csvfile,delimiter=',')

    labels = False

    for line in csvfile:

        if labels==False:

            labels = True

            continue

        else:

            train_label.append(int(line[0]))

            temp = []

            for pixel in line[1:]:

                temp.append(int(pixel))

            train_data.append(np.reshape(temp,(28,28,1)))



with open(test_file)  as csvfile:

    csvfile = csv.reader(csvfile,delimiter=',')

    labels = False

    for line in csvfile:

        if labels==False:

            labels = True

            continue

        else:

            test_label.append(int(line[0]))

            temp = []

            for pixel in line[1:]:

                temp.append(int(pixel))

            test_data.append(np.reshape(temp,(28,28,1)))            



train_data = np.array(train_data)

train_label = np.array(train_label)



test_data = np.array(test_data)

test_label = np.array(test_label)
print('train data = ',train_data.shape)

print('train label = ',train_label.shape)



print('test data = ',test_data.shape)

print('test label = ',test_label.shape)
train_data = train_data/255.0

test_data = test_data/255.0



r = list(range(0,len(train_label)))



for index in list(range(0,len(train_label))):

    if train_label[index] == 24:

        train_label[index]=9



for index in range(0,len(test_label)):

    if test_label[index] == 24:

        test_label[index]=9

    

classes = len(set(train_label))
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten

from tensorflow.keras import regularizers

model = tf.keras.Sequential()

model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(256,(3,3),activation='relu'))

model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(1024,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(classes,activation='softmax'))

model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_data,train_label,epochs=50,batch_size=64)
model.evaluate(test_data,test_label)