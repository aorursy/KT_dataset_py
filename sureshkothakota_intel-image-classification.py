import numpy as np         

import os                  

import random 

import matplotlib.pyplot as plt
import cv2

import tensorflow.keras.models as Models
import tensorflow.keras.layers as Layers

import tensorflow.keras.activations as Actications

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizer

import tensorflow.keras.metrics as Metrics

import tensorflow.keras.utils as Utils

from keras.utils.vis_utils import model_to_dot

import os

import matplotlib.pyplot as plot

import cv2

import numpy as np

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix as CM

from random import randint

from IPython.display import SVG

import matplotlib.gridspec as gridspec
train='/kaggle/input/intel-image-classification/seg_train/seg_train'

test='/kaggle/input/intel-image-classification/seg_test/seg_test'
category=['street','forest','glacier','mountain','sea','buildings']
datadir='/kaggle/input/intel-image-classification/seg_train/seg_train'

category=['street','forest','glacier','mountain','sea','buildings']

for i in category:

    path=os.path.join(datadir,i)

    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array,cmap='gray')

        plt.show()

        break

    break  
IMG_SIZE=150

new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)

plt.imshow(new_array,cmap='gray')

plt.show()
training_data=[]

def create_training_data():

    for i in category:

        path=os.path.join(train,i)

        class_num=category.index(i)

        for img in os.listdir(path):

            try:

                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)

                training_data.append([new_array,class_num])

            except Exception as e:

                pass

create_training_data()
training_data
len(training_data)
import random

random.shuffle(training_data)
for sample in training_data[:10]:

    print(sample[1])
testing_data=[]

def create_testing_data():

    for i in category:

        path=os.path.join(test,i)

        class_num=category.index(i)

        for img in os.listdir(path):

            try:

                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)

                testing_data.append([new_array,class_num])

            except Exception as e:

                pass

create_testing_data()
testing_data
len(testing_data)
import random

random.shuffle(testing_data)
for sample in testing_data[:10]:

    print(sample[1])
X=[]

y=[]
for features,label in training_data:

    X.append(features)

    y.append(label)
len(X)
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

X.shape



X.shape[1:]
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import tensorflow as tf  
model=Sequential()

model.add(Conv2D(256,(3,3),input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(128,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(625))

model.add(Activation('relu'))

model.add(Dense(6))

model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

X=X/255.0
X
X.shape
y=np.array(y,dtype=np.uint8)

y
y.shape
model.fit(X,y,batch_size=128,validation_split=0.35,epochs=30)
def prepare(filepath):

    img_size=150

    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

    new_array=cv2.resize(img_array,(img_size,img_size),interpolation=cv2.INTER_AREA)

    return np.array(new_array).reshape(-1,img_size,img_size,1)
a=prepare('/kaggle/input/intel-image-classification/seg_test/seg_test/street/20084.jpg')
pred=model.predict([a])

pred
model.predict_classes([a])
b=prepare('/kaggle/input/intel-image-classification/seg_test/seg_test/mountain/20058.jpg')

b=b/255.0

b
model.predict([b])
model.predict_classes([b])
len(testing_data)
x1=[]

y1=[]
for features,label in testing_data:

    x1.append(features)

    y1.append(label)
x1=np.array(x1).reshape(-1,IMG_SIZE,IMG_SIZE,1)

x1
y1=np.array(y1,dtype=np.uint8)

y1
x1=x1/255.0
model.predict_classes(x1)
y1[:5]
model.predict_classes(x1[:5])
test_loss = model.evaluate(X,y)