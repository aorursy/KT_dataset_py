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
import os
path='../input/chest-xray-for-covid19-detection/Dataset'
train_dir='../input/chest-xray-for-covid19-detection/Dataset/Train'
test_dir='../input/chest-xray-for-covid19-detection/Dataset/Val'
print(len(os.listdir(train_dir)))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
path_1='../input/chest-xray-for-covid19-detection/Dataset/Train/Covid/'
img=path_1+'1-s2.0-S0929664620300449-gr2_lrg-b.jpg'
img=mpimg.imread(img)
plt.imshow(img,cmap='gray')
img_2='../input/chest-xray-for-covid19-detection/Dataset/Train/Normal/IM-0140-0001.jpeg'
img_2=mpimg.imread(img_2)
print(img_2.shape)
plt.imshow(img_2,cmap="gray")

import cv2
def re_size(x):
    img=mpimg.imread(x)
    resized_image=cv2.resize(img,(150,150,3))
    return(resized_img)
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen=ImageDataGenerator(rescale=1/255,rotation_range=20,
                               width_shift_range=0.10,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest')
valid_gen=ImageDataGenerator(rescale=1./255)
img_width=150
img_height=150
ch=3
image_shape=(img_height,img_width,ch)
train=train_gen.flow_from_directory(train_dir,target_size=[150,150],color_mode='rgb',batch_size=16,class_mode='binary',shuffle=True)
valid=valid_gen.flow_from_directory(test_dir,target_size=[150,150],color_mode='rgb',batch_size=16,class_mode='binary',shuffle=True)
import tensorflow as tf
model=tf.keras.Sequential([
                          tf.keras.layers.Conv2D(32,(3,3),input_shape=image_shape,activation='relu'),
                          tf.keras.layers.MaxPooling2D((2,2)),
                          tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                          tf.keras.layers.MaxPooling2D((2,2)),
                          tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                          tf.keras.layers.MaxPooling2D((2,2)),
                          tf.keras.layers.Flatten(),
                          tf.keras.layers.Dense(128,activation='relu'),
                          tf.keras.layers.Dense(1,activation='sigmoid')])
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)


epochs=15
history=model.fit(train,epochs=15,validation_data=valid,callbacks=[early_stop])
prediction=model.predict(valid)
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(acc,label='Accuracy')
plt.plot(val_acc,label='loss')
plt.legend()
plt.figure(num=2)
plt.title('loss')
plt.plot(loss,label='loss')
plt.plot(val_loss,label='val_loss')
