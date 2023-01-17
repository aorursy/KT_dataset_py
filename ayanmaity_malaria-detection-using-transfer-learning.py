# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images"))
# Any results you write to the current directory are saved as output.
print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/Parasitized"))
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,imread
import random
para_img = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/Parasitized")
rand_img = random.sample(para_img,5)
plt.figure(1,figsize=(15,9))
for i,img in enumerate(rand_img):
    plt.subplot(1,5,i+1)
    print(img)
    plt.imshow(imread("../input/cell-images-for-detecting-malaria/cell_images/Parasitized/"+img))
plt.show()
un_img = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/Uninfected")
rand_img = random.sample(un_img,5)
plt.figure(1,figsize=(15,9))
for i,img in enumerate(rand_img):
    plt.subplot(1,5,i+1)
    print(img)
    plt.imshow(imread("../input/cell-images-for-detecting-malaria/cell_images/Uninfected/"+img))
plt.show()
from skimage.transform import resize
data = []
rand_para = random.sample(para_img,3000)
for i,img in enumerate(rand_para):
    try :
        temp = resize(imread("../input/cell-images-for-detecting-malaria/cell_images/Parasitized/"+img),(112,112))
    except :
        continue
    data.append(temp) 
para_len = len(data)
rand_un = random.sample(un_img,3000)
for i,img in enumerate(rand_un):
    try :
        temp = resize(imread("../input/cell-images-for-detecting-malaria/cell_images/Uninfected/"+img),(112,112))
    except :
        continue
    data.append(temp) 
len(data)
label_1 = [1 for i in range(para_len)]
label_0 = [0 for i in range(len(data)-para_len)]
label = label_1+label_0
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(np.array(data),np.array(label),test_size=0.10,random_state = 12,shuffle=True)

import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,BatchNormalization,Dropout,Add,Input,Lambda,Concatenate,Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import keras.backend as K
from keras.optimizers import Adam
from keras.applications import VGG16,Xception,ResNet50,MobileNetV2,InceptionV3,VGG19,DenseNet121
"""model = Sequential()
model.add(Conv2D(filters=8,kernel_size=(5,5),padding='valid',activation='relu',input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
##model.add(BatchNormalization())
##model.add(Dropout(0.2))

model.add(Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
##model.add(BatchNormalization())
##model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='sigmoid'))"""
input_imgs  = Input((112,112,3))

base = ResNet50(include_top=False, input_shape=(112,112,3), pooling='max')
base.layers.pop()
base.layers[-1].outbound_nodes = []
base.outputs = [base.layers[-1].output]
base_last = base(input_imgs)
#base_last = base(input_imgs)
#flat_1 = Flatten(name='flat_3')(vgg_last)
#dense_1 = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='dense_1')(base_last)
#dense_1 = Dropout(0.5,name='drop_1')(dense_1)
conv_last = Conv2D(512,kernel_size=4,name='conv_last')(base_last)
conv_last = Reshape((-1,),name='conv_reshape')(conv_last)
conv_last = Dropout(0.25,name='drop_1')(conv_last)
dense_2 = Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01),name='dense_2')(conv_last)
dense_2 = Dropout(0.5,name='drop_2')(dense_2)
output = Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.01),name='output')(dense_2)

model = Model(inputs=input_imgs,outputs=output,name='CNN Model')
##from keras.utils import plot_model
##plot_model(model, to_file='model.png',show_shapes=T
from keras.utils import plot_model
plot_model(base, to_file='model.png',show_shapes=True)
##base.summary()
for layer in base.layers[:]:
    layer.trainable = False
    
    if layer.name.startswith('bn'):
        layer.call(layer.input, training=False)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_x,train_y,validation_split=0.05,batch_size=128,epochs=20,verbose=1)
model.evaluate(train_x,train_y)
model.evaluate(test_x,test_y)
model.save('model_incep.h5')