# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import numpy as np

import pandas as pd

import matplotlib as plt

from PIL import Image

import os

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
import cv2
data=[]

labels=[]

height = 30

width = 30

channels = 3

classes = 43

n_inputs = height * width*channels



for i in range(classes) :

    path = "../input/gtsrb-german-traffic-sign/Train/{0}/".format(i)

    print(path)

    Class=os.listdir(path)

    for a in Class:

        try:

            image=cv2.imread(path+a)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((height, width))

            data.append(np.array(size_image))

            labels.append(i)

        except AttributeError:

            print(" ")

            

Cells=np.array(data)

labels=np.array(labels)

s=np.arange(Cells.shape[0])

np.random.seed(43)

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]
(X_train,X_val) = Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]

X_train = X_train.astype('float32')/255

X_val= X_val.astype('float32')/255

(y_train,y_val) = labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]
y_train = to_categorical(y_train,43)

y_val = to_categorical(y_val,43)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy'])
epochs = 20

history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,

validation_data=(X_val, y_val))

#Predicting with the test data

y_test=pd.read_csv("../input/gtsrb-german-traffic-sign/Test.csv")

labels=y_test['Path'].values

y_test=y_test['ClassId'].values



data=[]



for f in labels:

    image=cv2.imread('../input/gtsrb-german-traffic-sign/Test/'+f.replace('Test/', ''))

    image_from_array = Image.fromarray(image, 'RGB')

    size_image = image_from_array.resize((height, width))

    data.append(np.array(size_image))



X_test=np.array(data)

X_test = X_test.astype('float32')/255 

pred = model.predict_classes(X_test)
#Accuracy with the test data

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)