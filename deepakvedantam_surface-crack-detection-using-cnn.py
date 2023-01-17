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
from tensorflow import keras
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2
from matplotlib import pyplot as plt
train=ImageDataGenerator(rescale=1/255,validation_split=0.05)
train_dataset=train.flow_from_directory('../input/surface-crack-detection',
                                       target_size=(150,150),
                                       batch_size=32,
                                       class_mode='binary',
                                       subset='training',           
                                       )
val_dataset=train.flow_from_directory('../input/surface-crack-detection',
                                       target_size=(150,150),
                                      batch_size=32,
                                       class_mode='binary',
                                       subset='validation',           
                                       )
train_dataset.class_indices
model=Sequential()
model.add(Conv2D(256,(1,1),input_shape=(128,128,3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(1,1),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(1,1),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(3,3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(3,3),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256,(1,1),activation='relu',kernel_initializer='he_uniform'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(128,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model.add(Dense(64,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model.add(Dense(32,activation='relu',kernel_initializer='he_uniform',kernel_regularizer='l2'))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(train_dataset,epochs=1,validation_data=val_dataset,batch_size=128)
img=image.load_img('../input/my-data1/nocrack1.jpg',target_size=(150,150))
plt.imshow(img)
img1=image.img_to_array(img)
img1=img1/255
img1=np.expand_dims(img1,[0])
print(img1.shape)
pred=model.predict(img1)
if(pred[0]>=0.5):
    print("Crack Detected")
else:
    print("No Crack Detected")
pred
