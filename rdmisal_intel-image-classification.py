import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input/seg_train/seg_train/"))
import matplotlib.image as img

import cv2

from PIL import Image

data=[]

labels=[]

street = os.listdir('../input/seg_train/seg_train/street/')

%time

for a in street:

    try:

        image = cv2.imread('../input/seg_train/seg_train/street/'+a)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        data.append(np.array(size_image))

        labels.append(0)

    except AttributeError:

        print("")

        

sea = os.listdir('../input/seg_train/seg_train/sea/')

%time

for b in sea:

    try:

        image = cv2.imread('../input/seg_train/seg_train/sea/'+b)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        data.append(np.array(size_image))

        labels.append(1)

    except AttributeError:

        print("")

        

glacier = os.listdir('../input/seg_train/seg_train/glacier/')

%time

for c in glacier:

    try:

        image = cv2.imread('../input/seg_train/seg_train/glacier/'+c)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        data.append(np.array(size_image))

        labels.append(2)

    except AttributeError:

        print("")

        

mountain = os.listdir('../input/seg_train/seg_train/mountain/')

%time

for d in mountain:

    try:

        image = cv2.imread('../input/seg_train/seg_train/mountain/'+d)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        data.append(np.array(size_image))

        labels.append(3)

    except AttributeError:

        print("")

        

        

buildings = os.listdir('../input/seg_train/seg_train/buildings/')

%time

for e in buildings:

    try:

        image = cv2.imread('../input/seg_train/seg_train/buildings/'+e)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        data.append(np.array(size_image))

        labels.append(4)

    except AttributeError:

        print("")

        

forest = os.listdir('../input/seg_train/seg_train/forest/')

%time

for f in forest:

    try:

        image = cv2.imread('../input/seg_train/seg_train/forest/'+f)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        data.append(np.array(size_image))

        labels.append(5)

    except AttributeError:

        print("")
x_train=np.array(data)

y_train=np.array(labels)

y_train.shape
x_train.shape
import matplotlib.image as img

import cv2

from PIL import Image

test=[]

prediction=[]

street = os.listdir('../input/seg_test/seg_test/street/')

%time

for a in street:

    try:

        image = cv2.imread('../input/seg_test/seg_test/street/'+a)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        test.append(np.array(size_image))

        prediction.append(0)

    except AttributeError:

        print("")

        

sea = os.listdir('../input/seg_test/seg_test/sea/')

%time

for b in sea:

    try:

        image = cv2.imread('../input/seg_test/seg_test/sea/'+b)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        test.append(np.array(size_image))

        prediction.append(1)

    except AttributeError:

        print("")

        

glacier = os.listdir('../input/seg_test/seg_test/glacier/')

%time

for c in glacier:

    try:

        image = cv2.imread('../input/seg_test/seg_test/glacier/'+c)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        test.append(np.array(size_image))

        prediction.append(2)

    except AttributeError:

        print("")

        

mountain = os.listdir('../input/seg_test/seg_test/mountain/')

%time

for d in mountain:

    try:

        image = cv2.imread('../input/seg_test/seg_test/mountain/'+d)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        test.append(np.array(size_image))

        prediction.append(3)

    except AttributeError:

        print("")

        

        

buildings = os.listdir('../input/seg_test/seg_test/buildings/')

%time

for e in buildings:

    try:

        image = cv2.imread('../input/seg_test/seg_test/buildings/'+e)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        test.append(np.array(size_image))

        prediction.append(4)

    except AttributeError:

        print("")

        

forest = os.listdir('../input/seg_test/seg_test/forest/')

%time

for f in forest:

    try:

        image = cv2.imread('../input/seg_test/seg_test/forest/'+f)

        image_to_array=Image.fromarray(image,"RGB")

        size_image=image_to_array.resize((50,50))

        test.append(np.array(size_image))

        prediction.append(5)

    except AttributeError:

        print("")
x_test=np.array(test)

y_test=np.array(prediction)

y_test.shape
import tensorflow as tf

import tensorflow.keras as keras

from keras.utils import to_categorical

no_classes=6

ytr=to_categorical(y_train,no_classes)

yte=to_categorical(y_test,no_classes)

from keras.models import Sequential

from keras.layers import  Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dense(6,activation="softmax"))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size=500

epochs=40

predict=model.fit(x_train,ytr,batch_size=batch_size,epochs=epochs)
accuracy = model.evaluate(x_test, yte, verbose=1)

print('Test_Accuracy:-', accuracy[1])