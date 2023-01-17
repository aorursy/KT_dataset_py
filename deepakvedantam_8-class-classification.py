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
import  numpy as np
from tensorflow import keras
from keras.layers import Dense,Conv2D,MaxPool2D,BatchNormalization,Flatten
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
train=ImageDataGenerator(rescale=1/255,validation_split=0.15)
train_dataset=train.flow_from_directory('../input/natural-images/natural_images',
                                        target_size=(64,64),
                                        class_mode='sparse',
                                         subset='training',                                 
                                        )
val_dataset=train.flow_from_directory('../input/natural-images/natural_images',
                                        target_size=(64,64),
                                        class_mode='sparse',           
                                        subset='validation',
                                           
                                     )
train_dataset.class_indices
val_dataset.class_indices
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(keras.layers.Dropout(0.30))
model.add(Flatten())
#model.add(Dense(512,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model.add(Dense(256,activation='relu',kernel_initializer='he_uniform'))
#model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
#model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
model.add(keras.layers.Dropout(0.50))
model.add(Dense(8,activation='softmax',kernel_initializer='glorot_uniform'))
print(model.summary())

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_dataset,epochs=1,validation_data=val_dataset,batch_size=128)

model.evaluate(val_dataset)

for dirname, _, filenames in os.walk('../input/cat-dataset/CAT_00/**.jpg'):
    ct+=1
    for filename in filenames:
        print(os.path.join(dirname, filename))
print(ct)
img=image.load_img('../input/natural-images/natural_images/fruit/fruit_0003.jpg',target_size=(64,64))
img
img1=image.img_to_array(img)
img1=img1/255
img1=np.expand_dims(img1,[0])
img1.shape
k=model.predict(img1)
print(np.argmax(k))
train_dataset.class_indices
