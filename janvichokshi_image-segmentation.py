!unzip '/kaggle/input/carvana-image-masking-challenge/train_hq.zip' -d './'

!unzip '/kaggle/input/carvana-image-masking-challenge/train_masks.zip' -d './'

!unzip '/kaggle/input/carvana-image-masking-challenge/train.zip' -d './'

!unzip '/kaggle/input/carvana-image-masking-challenge/train_masks.csv.zip' -d './'
!pip install git+https://github.com/tensorflow/examples.git

!pip install -U tfds-nightly

import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds

import numpy as np

import pandas as pd 

from IPython.display import clear_output

import matplotlib.pyplot as plt

import glob

from PIL import Image

from keras.preprocessing import image

import cv2
glob.glob('./*')
mask_csv=pd.read_csv('./train_masks.csv')
glob.glob('./train/*')[0]
fig=plt.figure(figsize=(15, 15))

img=cv2.imread('./train/d46244bc42ed_04.jpg')

mask=Image.open('./train_masks/d46244bc42ed_04_mask.gif')

files=[img,mask]

for i in range(len(files)):

    plt.subplot(1, 2 , i+1)

    plt.imshow(files[i])
img.shape,image.img_to_array(mask).shape
def preprocess_image(img,mask,train=True):

    input_img=cv2.resize(img,(128,128))/255.0

    input_mask=cv2.resize(mask,(128,128))

    return input_img,input_mask
def load_imgs(name):

    input_img=cv2.imread('./train/'+name+'.jpg')

    input_mask=image.img_to_array(Image.open('./train_masks/'+name+'_mask.gif'))

    input_img,input_mask=preprocess_image(input_img,input_mask)

    return input_img,input_mask
x_data=[]

y_data=[]

imgs_path=glob.glob('./train/*')

for i in range(len(imgs_path)):

    input_img,input_mask=load_imgs(imgs_path[i][8:-4])

    x_data.append(input_img)

    y_data.append(input_mask)
x_d=np.array(x_data)

y_d=np.array(y_data)

train_data=int((x_d.shape[0]*0.80))

x_train=x_d[:train_data]

y_train=y_d[:train_data]

x_test=x_d[train_data:]

y_test=y_d[train_data:]

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)



y_train=y_train[...,np.newaxis]

y_test=y_test[...,np.newaxis]

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
import keras
OUTPUT_CHANNELS = 2

base_model = keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)



# Use the activations of these layers

layer_names = [

    'block_1_expand_relu',   # 64x64

    'block_3_expand_relu',   # 32x32

    'block_6_expand_relu',   # 16x16

    'block_13_expand_relu',  # 8x8

    'block_16_project',      # 4x4

]

layers = [base_model.get_layer(name).output for name in layer_names]



# Create the feature extraction model

down_stack = keras.Model(inputs=base_model.input, outputs=layers)



down_stack.trainable = False
up_stack = [

    pix2pix.upsample(512, 3),  # 4x4 -> 8x8

    pix2pix.upsample(256, 3),  # 8x8 -> 16x16

    pix2pix.upsample(128, 3),  # 16x16 -> 32x32

    pix2pix.upsample(64, 3),   # 32x32 -> 64x64

]
def unet_model(output_channels):

    inputs = keras.layers.Input(shape=[128, 128, 3])

    x = inputs



  # Downsampling through the model

    skips = down_stack(x)

    x = skips[-1]

    skips = reversed(skips[:-1])



  # Upsampling and establishing the skip connections

    for up, skip in zip(up_stack, skips):

        x = up(x)

        concat = keras.layers.Concatenate()

        x = concat([x, skip])



  # This is the last layer of the model

    last = keras.layers.Conv2DTranspose(output_channels, 3, strides=2,padding='same')  #64x64 -> 128x128



    x = last(x)



    return keras.Model(inputs=inputs, outputs=x)
model = unet_model(OUTPUT_CHANNELS)

model.compile(optimizer='adam',

              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model_history = model.fit(x_train,y_train, epochs=1,validation_data=(x_test,y_test))
i=580

pred_mask = model.predict(x_test[i:i+1])

print(pred_mask.shape)

y_pred=np.argmax(pred_mask[0],-1)

y_pred=y_pred[...,np.newaxis]

print(y_pred.shape)

fig=plt.figure(figsize=(15, 15))

img=x_test[i]

true_mask=keras.preprocessing.image.array_to_img(y_test[i])

pred_mask=keras.preprocessing.image.array_to_img(y_pred)

files=[img,true_mask,pred_mask]

for i in range(len(files)):

    plt.subplot(1, len(files) , i+1)

    plt.imshow((files[i]))