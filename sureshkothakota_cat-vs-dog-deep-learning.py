import numpy as np         

import os                  

import random 

import matplotlib.pyplot as plt

 

import cv2

TRAIN_DIR = '/kaggle/input/cat-and-dog/training_set/training_set'

TEST_DIR = '/kaggle/input/cat-and-dog/test_set/test_set'

category=['cats','dogs']
datadir='/kaggle/input/cat-and-dog/training_set/training_set'

category=['dogs','cats']

for i in category:

    path=os.path.join(datadir,i)

    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

        plt.imshow(img_array,cmap='gray')

        plt.show()

        break

    break  
IMG_SIZE=60

new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)

plt.imshow(new_array,cmap='gray')

plt.show()
training_data=[]

def create_training_data():

    for i in category:

        path=os.path.join(TRAIN_DIR,i)

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

        path=os.path.join(TEST_DIR,i)

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
model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))

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

model.add(Dense(100))

model.add(Activation('sigmoid'))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
X=X/255.0
X
y=np.array(y,dtype=np.uint8)

y
y.shape
model.fit(X,y,validation_split=0.3,epochs=30)
def prepare(filepath):

    img_size=60

    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

    new_array=cv2.resize(img_array,(img_size,img_size),interpolation=cv2.INTER_AREA)

    return np.array(new_array).reshape(-1,img_size,img_size,1)
a=prepare('/kaggle/input/cat-and-dog/test_set/test_set/cats/cat.4005.jpg')

a=a/255

a
pred=model.predict([a])

pred
model.predict_classes([a])
b=prepare('/kaggle/input/cat-and-dog/test_set/test_set/dogs/dog.4008.jpg')

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