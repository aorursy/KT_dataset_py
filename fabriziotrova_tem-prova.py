## Imports

import os

import sys

import random



import numpy as np

import cv2

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint

import tensorflow as tf

from tensorflow import keras

from keras.optimizers import adam

from keras import backend as K



## Seeding 

seed = 2020

random.seed = seed

np.random.seed = seed

tf.seed = seed
img = cv2.imread('/kaggle/input/temtrain/train/10_B1_19000x/data/10_B1_19000x.tif')

mask = cv2.imread('/kaggle/input/temtrain/train/10_B1_19000x/masks/10_B1_19000x.tif_segmentation.tifnormalized.tif')

fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)

ax.imshow(img)

ax = fig.add_subplot(1, 2, 2)

ax.imshow(mask)



#print(np.unique(mask), mask.shape, img.shape)

mask = mask / 255

mask_cat = tf.keras.utils.to_categorical(mask,3)

print (mask_cat.shape)
def uniq(lst):

    last = object()

    for item in lst:

        if item == last:

            continue

        yield item

        last = item

        

def sort_deduplicate(l):

    return uniq(sorted(l, reverse=True))



lista = []

for i in range(1024):

    for j in range(1024):

        lista.append(mask[i,j,:].tolist())



palette_gen = sort_deduplicate(lista)

palette = []

for item in palette_gen:

    palette.append(item)

    

palette
one_hot_map = []

for colour in palette:

    class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)

    one_hot_map.append(class_map)

one_hot_map = tf.stack(one_hot_map, axis=-1)

one_hot_map = tf.cast(one_hot_map, tf.float32)



tf.compat.v1.disable_eager_execution()



with tf.compat.v1.Session() as sess:

    print (sess.run(one_hot_map))
class DataGen(keras.utils.Sequence):

    def __init__(self, ids, path, batch_size=8, image_size=256, data_path = 'data', masks_path = 'masks'):

        self.ids = ids

        self.path = path

        self.batch_size = batch_size

        self.image_size = image_size

        self.on_epoch_end()

        

    def __load__(self, id_name):

        ## Path

        image_path = os.path.join(self.path, id_name, "data", id_name) + '.tif'

        # (image_path)

        mask_path = os.path.join(self.path, id_name, "masks/")

        print (mask_path)

        

        ## Reading Image

        image = cv2.imread(image_path)

        image = cv2.resize(image, (self.image_size, self.image_size))

        

        #mask = np.zeros((self.image_size, self.image_size, 1))

        

        ## Reading Masks

        _mask_path = mask_path + id_name + '.tif_segmentation.tifnormalized.tif'

        _mask_image = cv2.imread(_mask_path, 0)

        _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size)) #128x128

        mask = _mask_image

        

        ## Normalizaing 

        image = image/255.0

        mask = mask/255.0

        

        mask_cat = tf.keras.utils.to_categorical(mask,3)

        

        return image, mask_cat

    

    def __getitem__(self, index):

        if(index+1)*self.batch_size > len(self.ids):

            self.batch_size = len(self.ids) - index*self.batch_size

        

        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]

        

        image = []

        mask  = []

        

        for id_name in files_batch:

            _img, _mask = self.__load__(id_name)

            image.append(_img)

            mask.append(_mask)

            

        image = np.array(image)

        mask  = np.array(mask)

        

        return image, mask

    

    def on_epoch_end(self):

        pass

    

    def __len__(self):

        return int(np.ceil(len(self.ids)/float(self.batch_size)))
image_size = 256

train_path = "/kaggle/input/temtrain/train/"

validation_path = "/kaggle/input/temtest/test/"

epochs = 50

batch_size = 4



## Training Ids

train_ids = next(os.walk(train_path))[1]

#print(train_ids)



## Validation Ids

valid_ids = next(os.walk(validation_path))[1]



valid_ids = valid_ids[:]

train_ids = train_ids[:]

gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)

x, y = gen.__getitem__(0)

print(x.shape, y.shape)

r = random.randint(0, len(x)-1)



fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)

ax.imshow(x[r])

ax = fig.add_subplot(1, 2, 2)

ax.imshow(y[r])

print(x.shape, y.shape)
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)

    c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    p = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(c)

    return c, p



def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):

    us = tf.keras.layers.UpSampling2D((2, 2))(x)

    concat = tf.keras.layers.Concatenate()([us, skip])

    c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)

    c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    return c



def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)

    c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    return c
def UNet():

    f = [16, 32, 64, 128, 256]

    inputs = keras.layers.Input((image_size, image_size, 3))

    

    p0 = inputs

    c1, p1 = down_block(p0, f[0]) #128 -> 64

    c2, p2 = down_block(p1, f[1]) #64 -> 32

    c3, p3 = down_block(p2, f[2]) #32 -> 16

    c4, p4 = down_block(p3, f[3]) #16->8

    

    bn = bottleneck(p4, f[4])

    

    u1 = up_block(bn, c4, f[3]) #8 -> 16

    u2 = up_block(u1, c3, f[2]) #16 -> 32

    u3 = up_block(u2, c2, f[1]) #32 -> 64

    u4 = up_block(u3, c1, f[0]) #64 -> 128

    

    outputs = keras.layers.Conv2D(3, (1, 1), padding="same", activation="softmax")(u4)

    model = keras.models.Model(inputs, outputs)

    return model

def dice_coef(y_true, y_pred, smooth=1):

    intersection = K.sum(K.abs(y_true*y_pred ), axis=-1)

    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1-dice_coef(y_true, y_pred)



model = UNet()

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()
train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)

valid_gen = DataGen(valid_ids, validation_path, image_size=image_size, batch_size=batch_size)



train_steps = len(train_ids)//batch_size

valid_steps = len(valid_ids)//batch_size



model_checkpoint = ModelCheckpoint('unet_tem.hdf5', monitor='loss',verbose=1, save_best_only=True)

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 

                    epochs=epochs, callbacks=[model_checkpoint])

#model.load_weights('unet_tem.hdf5')
from skimage import img_as_uint

import skimage.io as io

Sky = [128,128,128]

Building = [128,0,0]

Pole = [192,192,128]

Road = [128,64,128]

Pavement = [60,40,222]

Tree = [128,128,0]

SignSymbol = [192,128,128]

Fence = [64,64,128]

Car = [64,0,128]

Pedestrian = [64,64,0]

Bicyclist = [0,128,192]

Unlabelled = [0,0,0]



COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,

                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):

    for i,item in enumerate(npyfile):

        img = item / 255

        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_as_uint(img))
results = model.predict_generator(valid_gen,10,verbose=1)
x, y = valid_gen.__getitem__(3)

result = model.predict(x)
fig.subplots_adjust(hspace=0.4, wspace=0.4)



fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)

ax.imshow(y[0])



ax = fig.add_subplot(1, 2, 2)

ax.imshow(result[0])
img = cv2.imread('/kaggle/input/temtest/test/31_B1_19000x/data/31_B1_19000x.tif')

mask = cv2.imread('/kaggle/input/temtest/test/31_B1_19000x/masks/31_B1_19000x.tif_segmentation.tifnormalized.tif')

fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)

ax.imshow(mask)

ax = fig.add_subplot(1, 2, 2)

img = cv2.resize(img, (256, 256))

img = np.expand_dims(img, axis=0)

ax.imshow(model.predict(img))

#print((model.predict(img).shape))
fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)

ax.imshow(mask)

ax = fig.add_subplot(1, 2, 2)

img = cv2.resize(img, (256, 256))

img = np.expand_dims(img, axis=0)

ax.imshow(model.predict(img))
fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)



ax = fig.add_subplot(1, 2, 1)

ax.imshow(np.reshape(results[1], (image_size, image_size)), cmap="gray")
fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)



ax = fig.add_subplot(1, 2, 1)

ax.imshow(np.reshape(y[1], (image_size, image_size)), cmap="gray")



ax = fig.add_subplot(1, 2, 2)

ax.imshow(np.reshape(result[1], (image_size, image_size)), cmap="gray")
# NUOVO MODELLO

import numpy as np 

import os

import skimage.io as io

import skimage.transform as trans

import numpy as np

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras import backend as keras





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

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)



    model = Model(input = inputs, output = conv10)



    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics = ['accuracy'])

    

    #model.summary()



    if(pretrained_weights):

        model.load_weights(pretrained_weights)



    return model