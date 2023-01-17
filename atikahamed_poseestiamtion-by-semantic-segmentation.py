# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

print(os.listdir('/kaggle/input'))

# Any results you write to the current directory are saved as output.
#!/usr/bin/env python

# coding: utf-8



# In[63]:





import tensorflow as tf

import keras.backend as K

import keras

from keras.models import *

from keras.layers import *

#importing necessary files

import scipy.io as sio

import numpy as np

import os

import json

import cv2

from pprint import pprint

import matplotlib.pyplot as plt

from keras import losses

from keras.optimizers import *

from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split

import time

from keras.models import load_model

import random

import scipy.io





# In[60]:





IMAGE_WIDTH=240

IMAGE_HEIGHT=240

N_CHANNELS=3

EPOCHS=300

LOSS_FUNC='categorical_crossentropy'

OPT='rmsprop'

METRICS=['accuracy']

BATCH_SIZE=10

AMOUNT=.02

NUM_CLASSES=20

color_map=[

       [  0,   0,   0],

       [128,   0,   0],

       [255,   0,   0],

       [  0,  85,   0],

       [170,   0,  51],

       [255,  85,   0],

       [  0,   0,  85],

       [  0, 119, 221],

       [ 85,  85,   0],

       [  0,  85,  85],

       [ 85,  51,   0],

       [ 52,  86, 128],

       [  0, 128,   0],

       [  0,   0, 255],

       [ 51, 170, 221],

       [  0, 255, 255],

       [ 85, 255, 170],

       [170, 255,  85],

       [255, 255,   0],

       [255, 170,   0]]





# In[68]:





file_path_y='/kaggle/input/lip_dataset/lip_dataset/LIP/TrainVal_parsing_annotations/train_segmentations/'

file_path_x='/kaggle/input/lip_dataset/lip_dataset/LIP/train_val_images/train_images/'

img_names_x=os.listdir(file_path_x)

#print(img_names_x)

img_names_y=os.listdir(file_path_y)

#print(img_names_y)



file_path_y_val='/kaggle/input/lip_dataset/lip_dataset/LIP/TrainVal_parsing_annotations/val_segmentations/'

file_path_x_val='/kaggle/input/lip_dataset/lip_dataset/LIP/train_val_images/val_images/'

img_names_x_val=os.listdir(file_path_x)

#print(img_names_x_val)

img_names_y_val=os.listdir(file_path_y_val)

#print(img_names_y_val)







# In[69]:





print('amount of data to train ',int(len(img_names_y)*AMOUNT))

print('amount of data to validate ',int(len(img_names_y_val)*AMOUNT))





# In[70]:





one_label=np.empty(( IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES), dtype=np.float32)

one_label.shape[2]





# In[71]:





x_train=[]

y_train=[]





for i  in range(0,int(len(img_names_y)*AMOUNT)):

    if i%1000==0:

        print('processed ',i)

    cur_image=cv2.imread(file_path_x+img_names_y[i].split('.')[0]+'.jpg')

    if cur_image is None:

        continue

    cur_image=cv2.resize(cur_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)

    cur_image=np.array(cur_image)

    cur_image=cur_image.astype('float')

    x_train.append(cur_image)

    

    cur_label_image=cv2.imread(file_path_y+img_names_y[i])

    cur_label_image=cv2.cvtColor(cur_label_image, cv2.COLOR_BGR2GRAY)

    cur_label_image=cv2.resize(cur_label_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)

    temp_label=np.empty(( IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES), dtype=np.float32)

    for x_label in range(cur_label_image.shape[0]):

        for y_label in range(cur_label_image.shape[1]):

            temp_label[x_label,y_label,cur_label_image[x_label,y_label]]=1

    

    y_train.append(temp_label)





# In[72]:





temp_label=np.empty(( IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES), dtype=np.float32)

for y_label in range(cur_label_image.shape[1]):

            y_label





# In[73]:





x_val=[]

y_val=[]





for i  in range(0,int(len(img_names_y_val)*AMOUNT)):

    if i%1000==0:

        print('processed ',i)

    cur_image=cv2.imread(file_path_x_val+img_names_y_val[i].split('.')[0]+'.jpg')

    if cur_image is None:

        continue

    cur_image=cv2.resize(cur_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)

    cur_image=np.array(cur_image)

    cur_image=cur_image.astype('float')

    x_val.append(cur_image)

    

    cur_label_image=cv2.imread(file_path_y_val+img_names_y_val[i])

    cur_label_image=cv2.cvtColor(cur_label_image, cv2.COLOR_BGR2GRAY)

    cur_label_image=cv2.resize(cur_label_image,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)

    temp_label=np.empty(( IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES), dtype=np.float32)

    for x_label in range(cur_label_image.shape[0]):

        for y_label in range(cur_label_image.shape[1]):

            temp_label[x_label,y_label,cur_label_image[x_label,y_label]]=1



    y_val.append(temp_label)





# In[74]:





def visualization(img):

    img=np.argmax(img,axis=2)

    #print(img.shape)

    ret = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS), np.float32)

    for r in range(IMAGE_WIDTH):

        for c in range(IMAGE_HEIGHT):

            color_id = img[r, c]

            # print("color_id: " + str(color_id))

            ret[r, c, :] = color_map[color_id]

    ret = ret.astype(np.uint8)

    #print(ret)

    plt.imshow(ret)

    plt.figure()





# In[75]:





rand_index=random.randint(0,int(len(img_names_y)*AMOUNT))

single_image=x_train[rand_index]

single_image=single_image.astype(np.uint8)

plt.imshow(single_image)



# In[76]:





single_image=y_train[rand_index]

print(np.argmax(single_image,axis=2))

visualization(single_image)





# In[77]:





rand_index=random.randint(0,int(len(img_names_y_val)*AMOUNT))

single_image=x_val[rand_index]

single_image=single_image.astype(np.uint8)

plt.imshow(single_image)





# In[78]:





single_image=y_val[rand_index]

visualization(single_image)





# In[79]:





x_train=np.array(x_train)

x_val=np.array(x_val)



y_train=np.array(y_train)

y_val=np.array(y_val)



print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)





# In[80]:





x_train=x_train/255.0

x_val=x_val/255.0





# In[81]:






def set_gpu_config(device = "0",fraction=1):

    config = tf.ConfigProto()

    config.gpu_options.per_process_gpu_memory_fraction = fraction

    config.gpu_options.visible_device_list = device

    K.set_session(tf.Session(config=config))





# In[82]:





class ODEBlock(Layer):



    def __init__(self, filters, kernel_size, **kwargs):

        self.filters = filters

        self.kernel_size = kernel_size

        super(ODEBlock, self).__init__(**kwargs)



    def build(self, input_shape):

        self.conv2d_w1 = self.add_weight("conv2d_w1", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')

        self.conv2d_w2 = self.add_weight("conv2d_w2", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')

        self.conv2d_b1 = self.add_weight("conv2d_b1", (self.filters,), initializer='zero')

        self.conv2d_b2 = self.add_weight("conv2d_b2", (self.filters,), initializer='zero')

        super(ODEBlock, self).build(input_shape)



    def call(self, x):

        t = K.constant([0, 1], dtype="float32")

        return tf.contrib.integrate.odeint(self.ode_func, x, t, rtol=1e-3, atol=1e-3)[1]



    def compute_output_shape(self, input_shape):

        return input_shape



    def ode_func(self, x, t):

        y = self.concat_t(x, t)

        y = K.conv2d(y, self.conv2d_w1, padding="same")

        y = K.bias_add(y, self.conv2d_b1)

        y = K.relu(y)



        y = self.concat_t(y, t)

        y = K.conv2d(y, self.conv2d_w2, padding="same")

        y = K.bias_add(y, self.conv2d_b2)

        y = K.relu(y)



        return y



    def concat_t(self, x, t):

        new_shape = tf.concat(

            [

                tf.shape(x)[:-1],

                tf.constant([1],dtype="int32",shape=(1,))

            ], axis=0)



        t = tf.ones(shape=new_shape) * tf.reshape(t, (1, 1, 1, 1))

        return tf.concat([x, t], axis=-1)





# In[51]:





def build_model(input_shape, num_classes):

    x = Input(input_shape)

    y = Conv2D(16, (3, 3), activation='relu',padding='same')(x)

    y = Conv2D(16, (3, 3), activation='relu',padding='same')(y)

    y = Conv2D(16, (3, 3), activation='relu',padding='same')(y)

    y = Conv2D(16, (3, 3), activation='relu',padding='same')(y)

    y = Conv2D(16, (3, 3), activation='relu',padding='same')(y)

    y = Conv2D(16, (3, 3), activation='relu',padding='same')(y)

    y = ODEBlock(16, (3, 3))(y)

    y = Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(y)

    return Model(x,y)



model = build_model((IMAGE_WIDTH,IMAGE_HEIGHT,N_CHANNELS), NUM_CLASSES)

model.summary()





# In[52]:





K.image_data_format()





# In[53]:





model.compile(loss=LOSS_FUNC,

              optimizer=keras.optimizers.Adam(),

              metrics=METRICS)





# In[54]:





#set_gpu_config("0",1)





# In[55]:





history=model.fit(x_train, y_train,

          batch_size=BATCH_SIZE,

          epochs=EPOCHS,

          verbose=1,validation_data=(x_val,y_val))
def unet(pretrained_weights = None,input_size = (256,256,1)):

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)



    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)



    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

    merge6 = concatenate([drop4,up6], axis = 3)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)



    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    merge7 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)



    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    merge8 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)



    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    merge9 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(20,(1, 1), activation = 'softmax')(conv9)



    model = Model(input = inputs, output = conv10)



    model.compile(optimizer = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=5E-6), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    

    #model.summary()



    if(pretrained_weights):

    	model.load_weights(pretrained_weights)



    return model

unet_model=unet(input_size=(IMAGE_WIDTH,IMAGE_HEIGHT,3))

unet_model.summary()
history=unet_model.fit(x_train, y_train,

          batch_size=BATCH_SIZE,

          epochs=EPOCHS,

          verbose=1,validation_data=(x_val,y_val))
test_img=cv2.imread('/kaggle/input/lip_dataset/lip_dataset/LIP/997_10.jpg')

plt.imshow(test_img)

x_test=[]

test_img=cv2.resize(test_img,(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)

x_test.append(test_img)

x_test=np.array(x_test)

x_test=x_test.astype('float')/255.0

print(x_test.shape)



y_pred=unet_model.predict(x_test)

y_pred=np.reshape(y_pred,(IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CLASSES))
visualization(y_pred)
unet_model.save('unet_model.h5') 