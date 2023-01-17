import numpy as np

import random 

import keras

from keras import backend as K



import cv2

import os

from glob import glob

import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf

seed = 2109

random.seed = seed

np.random.seed = seed

tf.seed = seed

all_images = glob(os.path.join('../input', '2d_images', '*.tif'))

all_masks = ['_masks'.join(c_file.split('_images')) for c_file in all_images]

class DataGen(keras.utils.Sequence):

    def __init__(self,train_images,train_masks,batch_size=8,image_size=128):

        self.train_images = train_images

        self.train_masks =train_masks

        self.image_size = image_size

        self.batch_size = batch_size 

        self.on_epoch_end

    

    def __load__(self,image_path,mask_path):

        image_path = image_path

        mask_path = mask_path

        image = cv2.imread(image_path,1)

        image = cv2.resize(image,(self.image_size,self.image_size))

        

        mask = cv2.imread(mask_path,0)

        mask = cv2.resize(mask,(self.image_size,self.image_size))

        mask = np.expand_dims(mask,axis = -1)



            

        image = image/255

        mask = mask/255  

        return image,mask

    def __getitem__(self, index):

        

        if(index+1)*self.batch_size > len(self.train_images):

            self.batch_size = len(self.train_images)- index*self.batch_size

        image_batch = self.train_images[index*self.batch_size : (index+1)*self.batch_size]

        mask_batch = self.train_masks[index*self.batch_size : (index+1)*self.batch_size]

        

        image = []

        mask = []

        for i, j in zip(image_batch, mask_batch):

            _img,_mask =self.__load__(i,j)

            image.append(_img)                

            mask.append(_mask)

        image = np.array(image)

        mask = np.array(mask)



        return image,mask

    def on_epoch_end(self):

        pass

    def __len__(self):

        return int(np.ceil(len(self.train_images)/float(self.batch_size)))



train_images = all_images

train_masks = all_masks

epochs = 5

valid_images = train_images[:30]

training_images = train_images[30:]

valid_masks = train_masks[:30]

training_masks = train_masks[30:]

gen = DataGen(training_images,training_masks,batch_size = 8,image_size=128)

x,y = gen.__getitem__(1)



r = random.randint(0, len(x)-1)





fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)



fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

ax2.imshow(np.reshape(y[r],(128,128)), cmap= "gray")

ax1.imshow(x[r])



def Dilated_Spatial_Pyramid_Pooling(x,k):

    x =  keras.layers.BatchNormalization(axis = 3)(x)

    d1 = keras.layers.Conv2D(k, (1,1), dilation_rate = 2)(x)

    d2 = keras.layers.Conv2D(k, (1,1), dilation_rate = 4)(d1)

    d3 = keras.layers.Conv2D(k, (1,1), dilation_rate = 8)(d2)

    d4 = keras.layers.Conv2D(k, (1,1), dilation_rate = 16)(d3)

    c = keras.layers.Concatenate()([d1,d2,d3,d4])

    return c

def down_block(x,filters, kernel_size = (3, 3), padding = "same",strides =1 ):

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(x)

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(c)

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(c)



    p = keras.layers.MaxPool2D((2,2),(2,2))(c)

    return c,p

def up_block(x,skip,filters, kernel_size = (3, 3), padding = "same",strides =1 ):

    us = keras.layers.UpSampling2D((2,2))(x)

    concat = keras.layers.Concatenate()([us,skip])

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(concat)

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(c)

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(c)

    return c

def bottleneck(x,filters, kernel_size = (3, 3), padding = "same",strides =1 ):

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(x)

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(c)

    c = keras.layers.Conv2D(filters, kernel_size,padding=padding,strides = strides, activation = "relu")(c)

    c = Dilated_Spatial_Pyramid_Pooling(c,filters)



    return c



def UNet():

    f = [16,32,64,128,256]

    input = keras.layers.Input((128,128,3))

    

    p0 = input

    c1,p1 =  down_block(p0,f[0])

    c2,p2 =  down_block(p1,f[1])

    c3,p3 =  down_block(p2,f[2])

    c4,p4 =  down_block(p3,f[3])

    

    bn = bottleneck(p4,f[4])

    

    u1 = up_block(bn,c4,f[3])

    u2 = up_block(u1,c3,f[2])

    u3 = up_block(u2,c2,f[1])

    u4 = up_block(u3,c1,f[0])

    

    

    outputs = keras.layers.Conv2D(1,(1,1),padding= "same",activation = "sigmoid")(u4)

    model = keras.models.Model(input ,outputs)

    return model

model = UNet()

model.compile(optimizer = "adam",loss = "binary_crossentropy", metrics = ["acc"])

model.summary()

train_gen = DataGen(training_images,training_masks,batch_size = 8, image_size = 128)

valid_gen = DataGen(valid_images,valid_masks,batch_size = 8, image_size = 128)

train_steps = len(training_images)//8

valid_steps = len(valid_images)//8



model.fit_generator(train_gen,validation_data = valid_gen,steps_per_epoch=train_steps,validation_steps =valid_steps,epochs= 10)



x,y = valid_gen.__getitem__(3)

result = model.predict(x)

result = result >0.5



fig = plt.figure()

fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

ax1 = fig.add_subplot(1,2,1)

ax1.imshow(np.reshape(y[0]*255,(128,128)), cmap= "gray")

ax2 = fig.add_subplot(1,2,2)

ax2.imshow(np.reshape(result[0]*255,(128,128)), cmap= "gray")
