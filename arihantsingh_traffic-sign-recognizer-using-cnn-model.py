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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import cv2

import os

from PIL import Image
data=[]

labels=[]

height = 30

width = 30

channels = 3

classes = 43

n_inputs = height * width*channels

for i in range(classes):

    path = "../input/gtsrb-german-traffic-sign/train/{0}".format(i) 

    print(path)

    Class=os.listdir(path)

    

    for a in Class:

        try :

            image=cv2.imread((path+"/{0}").format(a))

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((height, width))

            data.append(np.array(size_image))

            labels.append(i)

        except AttributeError:

            print("X")

                





print(type(labels))

Cells=np.array(data)

labels=np.array(labels)

print(len(Cells))
s=np.arange(len(Cells))

np.random.seed(43)

np.random.shuffle(s)







Cells=Cells[s]



labels=labels[s]
X_train=Cells[int(0.2*len(labels)): ]

X_val=Cells[ : int(0.2*len(labels))]

s=np.amax(Cells)

X_train=X_train.astype('float32')/s

X_val=X_val.astype('float32')/s

y_train=labels[(int)(0.2*len(labels)):]

y_val=labels[:(int)(0.2*len(labels))]

print(y_train)
from keras.utils import to_categorical

s1=np.amax(labels)

y_train=to_categorical(y_train,s1+1)

y_val=to_categorical(y_val, s1+1)

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model= Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(43, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=20,batch_size=32, validation_data=(X_val, y_val))
print(history)

import matplotlib.pyplot as plt

plt.figure(0)

plt.plot(history.history['accuracy'], label='training accuracy')

plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()
plt.figure(1)

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
#Predicting with the test data

y_test=pd.read_csv("../input/gtsrb-german-traffic-sign/Test.csv")

print(y_test.head())

labels=y_test['Path'].as_matrix()

y_test=y_test['ClassId'].values

print(labels)
d=[]

for j in labels:

    i1=cv2.imread('../input/gtsrb-german-traffic-sign/'+j)

    i2=Image.fromarray(i1,'RGB')

    i3=i2.resize((height,width))

    d.append(np.array(i3))

    
X_test=np.array(d)

X_test = X_test.astype('float32')/255 

pred = model.predict_classes(X_test)
#Accuracy with the test data

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)