# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from io import BytesIO

import cv2

import bson

from skimage.data import imread

import matplotlib.pyplot as plt

import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
category_data=pd.read_csv("../input/category_names.csv")

print("Total categories are:", len(category_data))

category_data.head(0)
def get_the_data(path):

    data = bson.decode_file_iter(open(path, 'rb'))

    images=[]

    category=[]

    for c, d in enumerate(data):

        product_id = d['_id']

        category_id = d['category_id'] # This won't be in Test data

        #prod_to_category[product_id] = category_id

        for e, pic in enumerate(d['imgs']):

            category.append(category_id)

            picture = imread(BytesIO(pic['picture']))

            #picture=pic['picture']

            images.append(picture)

            #break

        if(len(set(category))==1500):

            break

    return category, images
product_category_train,image_train=get_the_data('../input/train_example.bson')
product_category_train,image_train=get_the_data('../input/train.bson')
def img2feat(im):

    return np.float32(im) / 255
final=np.array(image_train)
final_train=img2feat(final)
y, rev_labels = pd.factorize(product_category_train)
from sklearn.utils import shuffle

im_train,lab_train=shuffle(final_train,y)

test_im=im_train[20000:]

test_lab=lab_train[20000:]

image_train=im_train[:20000]

label_train=lab_train[:20000]
from keras.layers import Conv2D, MaxPooling2D,Dropout,Dense, Flatten

from keras.models import Sequential

from keras.optimizers import Adam
model=Sequential()

model.add(Conv2D(16,3,activation='relu',input_shape=(180,180,3)))

model.add(Conv2D(32,3,activation='relu'))

model.add(MaxPooling2D(2))

model.add(Dropout(0.2))

model.add(Conv2D(32,3,activation='relu'))

model.add(Conv2D(32,3,activation='relu'))

model.add(MaxPooling2D(2))

model.add(Dropout(0.2))

model.add(Conv2D(64,3,activation='relu'))

model.add(Conv2D(64,3,activation='relu'))

model.add(MaxPooling2D(2))

model.add(Dropout(0.2))

model.add(Conv2D(32,3,activation='relu'))

model.add(Conv2D(16,3,activation='relu'))

model.add(MaxPooling2D(2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(len(rev_labels),activation='softmax'))
model.compile('Adam','sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(final_train,y,validation_split=0.2,epochs=2,batch_size=50)
test=np.array(image)

test_image=np.float32(test)/255

pred=model.predict(test_image)
acc=[]

for i in pred:

    acc.append(np.argmax(i))
rev_labels[acc[104]]
product_category[104]
label_acc=[]

for i in acc:

    label_acc.append(rev_labels[i])
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(product_category,label_acc)
accuracy