# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
train_data='/kaggle/input/natural-images/natural_images/'

import tensorflow as tf

import os

import glob
label_list=['airplane/','car/','cat/','dog/','flower/','fruit/','motorbike/','person/']

label=[0,1,2,3,4,5,6,7]

train=[]

for j,k in zip(label_list,label):

    for i in glob.glob(train_data+j+'*.jpg'):

        

        train.append(i)

        label.append(k)

        
label=label[8:]

len(label)
from keras.utils import to_categorical

label=to_categorical(label)
from sklearn.model_selection import train_test_split

tr_x,te_x,tr_y,te_y=train_test_split(train,label,train_size=0.7,random_state=4000)
data_train=tf.data.Dataset.from_tensor_slices((tr_x,tr_y))

data_test=tf.data.Dataset.from_tensor_slices((te_x,te_y))
def decode_img(img,label=None):

    img=tf.io.read_file(img)

    img=tf.image.decode_jpeg(img)

    img=tf.image.resize(img,[150,150])

    img=tf.cast(img,tf.float32)/255.0

    return img,label

    

data_train=data_train.map(decode_img)

data_test=data_test.map(decode_img)
auto=tf.data.experimental.AUTOTUNE

data_train=data_train.shuffle(buffer_size=len(tr_x))

data_train=data_train.repeat()

data_train=data_train.batch(32)

data_train=data_train.prefetch(auto)









data_test=data_test.shuffle(buffer_size=len(te_x))

data_test=data_test.repeat()

data_test=data_test.batch(32)

data_test=data_test.prefetch(auto)
data_train=data_train.as_numpy_iterator()

data_test=data_test.as_numpy_iterator()
!pip install -q efficientnet
import efficientnet.tfkeras as efn

import tensorflow.keras.layers as lr
from keras import regularizers
model=tf.keras.Sequential([efn.EfficientNetB7( input_shape=(150, 150, 3),weights='imagenet',include_top=False),

                           lr.GlobalAveragePooling2D(),

                           lr.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu'),

                           lr.Dropout(0.5),

                           lr.Dense(8, activation='softmax')])

model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

model.summary()
history=model.fit_generator(data_train,steps_per_epoch=len(tr_x)//32,epochs=10,validation_data=data_test,validation_steps=len(te_x)//32)
import matplotlib.pyplot as plt

hist=history.history

loss_values=hist['loss']

val_loss_values=hist['val_loss']



epochs=range(10)

plt.plot(epochs,loss_values,'bo',label='Train loss')

plt.plot(epochs,val_loss_values,'b',label='Validation loss')

plt.legend()

plt.show()





plt.clf()

acc_val=hist['accuracy']

val_acc_val=hist['val_accuracy']

plt.plot(epochs,acc_val,'bo',label='Train acc')

plt.plot(epochs,val_acc_val,'b',label='Validation acc')

plt.legend()

plt.show()