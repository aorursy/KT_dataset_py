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



# Any results you write to the current directory are saved as output.
import os

from os.path import join

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

import random

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras import backend as keras

from keras import utils as np_utils



#Imprimimos las carpetas que tenemos en nuestros datos

print(os.listdir("../input/"))

#Mascaras, porcentajes de las cosas dentro de las imagenes y las imagenes en si.
IMG_SIZE = 256

batch_size=30

epochs=5

images_path = "../input/final-dataset/Training_dataset/Images/"

masks_path = "../input/final-dataset/Training_dataset/Masks/all/"

images = os.listdir(images_path)

ordered_masks = []

for item in images:

    ordered_masks.append(item)

    

print(ordered_masks[0])

print(images[0])
lista_imagenes = os.listdir(images_path)

lista_mascaras = os.listdir(masks_path)



num_random = random.randint(0,len(lista_imagenes))

random_img = images[num_random] 

print(images_path+"/"+random_img)

random_mask = ordered_masks[num_random]

imagen = cv2.imread(images_path+"/"+random_img)

mascara = cv2.imread(masks_path+"/"+random_mask)

mascara=cv2.cvtColor(mascara, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))

fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)

ax1.imshow(imagen)

ax2 = fig.add_subplot(2,2,2)

ax2.imshow(mascara)

class DataGen(np_utils.Sequence):

    def __init__(self,images,images_path,mask_path,batch_size=16,image_size=256):

        self.batch_size=batch_size

        self.images = images

        self.image_size=image_size

        self.images_path = images_path

        self.mask_path = mask_path

        self.on_epoch_end()    

        

    def __load__(self, img):

        image_path = os.path.join(self.images_path+"/"+img)

        mask_path = os.path.join(self.mask_path+"/"+img)

        

        

        image = cv2.imread(image_path)

        image = cv2.resize(image,(self.image_size,self.image_size))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path,0)

        mask = cv2.resize(mask,(self.image_size,self.image_size))

        #mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)

        mask = np.expand_dims(mask,axis=-1)



        image = image/255.0

        mask = mask/255.0



        return image,mask

    

    def __getitem__(self, index):

        if(index+1)*self.batch_size>len(self.images):

            self.batch_size = len(self.images)- index*self.batch_size

            

        files_batch = self.images[index*self.batch_size : (index+1)*self.batch_size]



        image=[]

        mask=[]

        angle=90

        scale=1.0



        for imagen in files_batch:

            _img,_mask = self.__load__(imagen)

            image.append(_img)

            mask.append(_mask)

        image = np.array(image)

        mask = np.array(mask)



        return image,mask



    def on_epoch_end(self):

        pass

    def __len__(self):

        return int(np.ceil(len(self.images)/float(self.batch_size)))
data = os.listdir(images_path)

val_data_size = 1002

X_train = data[val_data_size:]

X_val = data[:val_data_size]

print(len(X_train))
image_size=256

batch_size=16

gen = DataGen(X_train,images_path, masks_path, batch_size=batch_size,image_size=image_size)

print(gen)

X,y = gen.__getitem__(0)

print(X.shape , y.shape)
r = random.randint(0,len(X-1))

fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)

ax.imshow(X[r])









#https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb
def bn_act(x, act=True):

    x = BatchNormalization()(x)

    if act == True:

        x = Activation("relu")(x)

    return x



def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    conv = bn_act(x)

    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

    return conv



def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)

    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)

    shortcut = bn_act(shortcut, act=False)

    

    output = Add()([conv, shortcut])

    return output



def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)

    shortcut = bn_act(shortcut, act=False)

    

    output = Add()([shortcut, res])

    return output



def upsample_concat_block(x, xskip):

    u = UpSampling2D((2, 2))(x)

    c = Concatenate()([u, xskip])

    return c
def ResUNet():

    f = [16, 32, 64, 128, 256]

    #    inputs = Input((256, 256, 3))

    inputs = Input((256, 256, 3))

    

    ## Encoder

    e0 = inputs

    e1 = stem(e0, f[0])

    e2 = residual_block(e1, f[1], strides=2)

    e3 = residual_block(e2, f[2], strides=2)

    e4 = residual_block(e3, f[3], strides=2)

    e5 = residual_block(e4, f[4], strides=2)

    

    ## Bridge

    b0 = conv_block(e5, f[4], strides=1)

    b1 = conv_block(b0, f[4], strides=1)

        

    ## Decoder

    u1 = upsample_concat_block(b1, e4)

    d1 = residual_block(u1, f[4])

    

    u2 = upsample_concat_block(d1, e3)

    d2 = residual_block(u2, f[3])

    

    u3 = upsample_concat_block(d2, e2)

    d3 = residual_block(u3, f[2])

    

    u4 = upsample_concat_block(d3, e1)

    d4 = residual_block(u4, f[1])

    

    #outputs = Conv2D(3, (1, 1), padding="same", activation="softmax")(d4)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs)

    return model
model = ResUNet()

adam =Adam()

#model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["acc"])



model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["acc"])

model.summary()
print(len(X_train))
#gen = DataGen(X_train,images_path, masks_path, batch_size=batch_size,image_size=image_size)



train_gen = DataGen(X_train,images_path,masks_path,batch_size=batch_size, image_size=image_size)

valid_gen  = DataGen(X_val,images_path,masks_path,batch_size=batch_size,image_size=image_size)



train_steps = len(X_train)//batch_size

valid_steps = len(X_val)//batch_size

print(train_steps)



model.fit_generator(train_gen,validation_data = valid_gen, steps_per_epoch=train_steps,

                   validation_steps = valid_steps, epochs=epochs)
model2 = ResUNet()

adam =Adam()

#model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["acc"])



model2.compile(optimizer=adam, loss="binary_crossentropy", metrics=["acc"])

model2.summary()



train_gen = DataGen(X_train,images_path,masks_path,batch_size=batch_size, image_size=image_size)

valid_gen  = DataGen(X_val,images_path,masks_path,batch_size=batch_size,image_size=image_size)



train_steps = len(X_train)//batch_size

valid_steps = len(X_val)//batch_size

print(train_steps)



model2.fit(train_gen,validation_data = valid_gen, steps_per_epoch=train_steps,

                   validation_steps = valid_steps, epochs=epochs)
history = model2.history

fig = plt.figure()

plt.subplot(2,1,1)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Precision de nuestro modelo")

plt.ylabel("Precision")

plt.xlabel("epochs")

plt.legend(["Train","Test"],loc="lower right")
plt.subplot(2,1,2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model loss")

plt.ylabel("Loss")

plt.xlabel("epochs")

plt.legend(["train","test"],loc="lower right")
x,y = valid_gen.__getitem__(3)

print(x.shape, y.shape)

#plt.imshow(x)

pred = model.predict(x)

print(pred.shape)



r = random.randint(0,len(x-1))



fig = plt.figure()

fig.subplots_adjust(hspace=0.6, wspace=0.6)



fig, (ax1,ax2,ax3)= plt.subplots(1,3)

#ax = fig.add_subplot(1, 2, 1)

ax1.imshow(x[r])

#ax = fig.add_subplot(1, 2, 2)

#pred[r] = cv2.resize(pred[r],(256,256))

print(pred[r].shape)

ax2.imshow(np.reshape(pred[r]*255, (256, 256)))

#ax = fig.add_subplot(1,3,1)

ax3.imshow(np.reshape(y[r]*255, (256, 256)))



for i in range(1,10,1):

    x,y = valid_gen.__getitem__(i)

    result = model.predict(x)

    #result = result >0.4

    r = random.randint(0,len(x-1))

    for i in range(len(result)):

        fig = plt.figure()

        fig.subplots_adjust(hspace=0.8,wspace=0.8)

        fig, (ax1,ax2,ax3)= plt.subplots(1,3)

        

        

        #result[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ax1.imshow(x[i])

        #ax.imshow(np.reshape(y[i]*255,(image_size,image_size)))

        ax2.imshow(np.reshape(y[i],(256,256)))

        

        ax3.imshow(np.reshape(result[i],(256,256)))

        #ax.imshow(np.reshape(result[i]*255,(image_size,image_size)))
#external_test_path = "../input/test-data"

#external_test_imgs = os.listdir(external_test_path)

#external_test_imgs = [os.path.join(external_test_path+"/"+img) for img in external_test_imgs]

#test_images = [cv2.imread(image)for image in external_test_imgs]

#test_images = np.array(test_images)

#image = cv2.imread(os.path.join(external_test_path+"/"+external_test_imgs[0]))

#image = cv2.resize(image,(256,256))

#image = np.expand_dims(image,axis=-1)

#external_predict = model.predict(image)