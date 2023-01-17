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
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten

from tensorflow.keras.models import Sequential

from matplotlib.image import imread

import matplotlib.pyplot as pyplt

import seaborn as sns

import os

import numpy as np
train_dir='/kaggle/input/intel-image-classification/seg_train/seg_train/'

train_street_dir=train_dir+'street/'



dim1=[]

dim2=[]

channels=[]



for img in os.listdir(train_street_dir):

    d1,d2,channel=imread(train_street_dir+img).shape

    dim1.append(d1)

    dim2.append(d2)

    channels.append(channel)

image_shape=(150,150,3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rescale=1/255,fill_mode='nearest',horizontal_flip=True,zoom_range=0.1)

train_data=image_gen.flow_from_directory(train_dir,target_size=image_shape[:2],color_mode='rgb',batch_size=16,shuffle=False,class_mode='binary')
train_data.class_indices
test_dir='/kaggle/input/intel-image-classification/seg_test/seg_test/'

test_data=image_gen.flow_from_directory(test_dir,target_size=image_shape[:2],color_mode='rgb',batch_size=16,shuffle=False,class_mode='binary')
from tensorflow.keras.layers import Activation,Conv2D,MaxPooling2D,Flatten,Dense,Dropout

from tensorflow.keras.models import Sequential



model=Sequential()
model.add(Conv2D(filters=32,activation='relu',input_shape=(150,150,3),kernel_size=(3,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64,activation='relu',input_shape=(150,150,3),kernel_size=(3,3)))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,activation='relu',input_shape=(150,150,3),kernel_size=(3,3)))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,activation='relu',input_shape=(150,150,3),kernel_size=(3,3)))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())





model.add(Dense(128))

model.add(Activation('relu'))



# model.add(Dense(128))

# model.add(Activation('relu'))



# Dropouts help reduce overfitting by randomly turning neurons off during training.

# Here we say randomly turn off 50% of neurons.

# model.add(Dropout(0.5))



model.add(Dense(6))

model.add(Activation('softmax'))



model.compile(loss='sparse_categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit_generator(train_data,epochs=20,validation_data=test_data)
model.metrics_names
model.evaluate_generator(test_data,verbose=1)
prd_prababilities=model.predict_generator(test_data)

prd_prababilities
test_data.classes
from tensorflow.keras.preprocessing import image
train_data.class_indices
labels={'buildings': 0,

 'forest': 1,

 'glacier': 2,

 'mountain': 3,

 'sea': 4,

 'street': 5}
def getClassNumber(className):

    for label in labels:

        if labels.get(label)==className:

            print(label)

            break
pred_dir='/kaggle/input/intel-image-classification/seg_pred/seg_pred/88.jpg'

actualImage=image.load_img(pred_dir)

final_test_image=image.img_to_array(actualImage).reshape(1,150,150,3)

getClassNumber(model.predict_classes(final_test_image))