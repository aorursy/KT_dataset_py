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
from __future__ import absolute_import, division, print_function

from PIL import Image

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.  

import os

print(os.listdir("../input/mias-jpeg/MIAS-JPEG/"))
infected = os.listdir('../input/mias-jpeg/MIAS-JPEG/abnormal/') 

uninfected = os.listdir('../input/mias-jpeg/MIAS-JPEG/normal/')


images =[]

classes=[]

for class_folder_name in os.listdir('../input/mias-jpeg/MIAS-JPEG/'):

    class_folder_path = os.path.join('../input/mias-jpeg/MIAS-JPEG/', class_folder_name)

    class_label = class_folder_name

    classes.append(class_label)


data=[]

labels=[]

Abnormal=os.listdir("../input/mias-jpeg/MIAS-JPEG/abnormal/")

for a in Abnormal:

    try:

        image=cv2.imread("../input/mias-jpeg/MIAS-JPEG/abnormal/"+a)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((224,224))

        data.append(np.array(size_image))

        labels.append(0)

    except AttributeError:

        print("")



Normal=os.listdir("../input/mias-jpeg/MIAS-JPEG/normal/")

for b in Normal:

    try:

        image=cv2.imread("../input/mias-jpeg/MIAS-JPEG/normal/"+b)

        image_from_array = Image.fromarray(image,"RGB")

        size_image = image_from_array.resize((224,224))

        data.append(np.array(size_image))

        labels.append(1)

    except AttributeError:

        print("")


Cells=np.array(data)

labels=np.array(labels)
np.save("Cells",Cells)

np.save("labels",labels)


Cells=np.load("Cells.npy")

labels=np.load("labels.npy")
print('Cells : {} | labels : {}'.format(Cells.shape , labels.shape))
plt.figure(1 , figsize = (22, 7))

n = 0 

for i in range(48):

    n += 1 

    r = np.random.randint(0 , Cells.shape[0] , 1)

    plt.subplot(7 , 7 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    plt.imshow(Cells[r[0]])

    plt.title('{} : {}'.format('Abnormal' if labels[r[0]] == 1 else 'Normal',labels[r[0]]) )

    plt.xticks([]) , plt.yticks([])

    

plt.show()
num_classes=len(np.unique(labels))

len_data=len(Cells)

plt.figure(1, figsize = (15 , 7))

plt.subplot(1 , 2 , 1)

plt.imshow(Cells[0])

plt.title('Infected Cell')

plt.xticks([]) , plt.yticks([])



plt.subplot(1 , 2 , 2)

plt.imshow(Cells[320])

plt.title('Uninfected Cell')

plt.xticks([]) , plt.yticks([])



plt.show()


s=np.arange(Cells.shape[0])

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]
(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]

x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.

x_test = x_test.astype('float32')/255

train_len=len(x_train)

test_len=len(x_test)
(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
y_train=keras.utils.to_categorical(y_train,num_classes)

y_test=keras.utils.to_categorical(y_test,num_classes)
#creating sequential model for single layer

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)


#accuracy of the model when single layer is used

accuracy = model.evaluate(x_test, y_test, verbose=1)

print('\n', 'Test_Accuracy:-', accuracy[1])

#Adding more layers to test the accuracy

#creating sequential model for 3 convultion layers

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()
# compile the model with loss as categorical_crossentropy and using adam optimizer you can test result by trying RMSProp as well as Momentum

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)
#accuracy when more number of layers are added

accuracy = model.evaluate(x_test, y_test, verbose=1)

print('\n', 'Test_Accuracy:-', accuracy[1])
model.history.history.keys()
plt.plot(model.history.history['acc'])

plt.plot(model.history.history['loss'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('loss')

plt.legend(['Accuracy', 'loss'], loc='upper right')

plt.show()