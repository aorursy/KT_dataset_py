import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
from keras.layers import *
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping,ReduceLROnPlateau
import keras.backend as K
from keras.preprocessing.image import img_to_array,load_img
TRAIN_IMG_DIR='../input/cityscapes_data/cityscapes_data/train/'
VALID_IMG_DIR='../input/cityscapes_data/cityscapes_data/val/'
IMG_ROW=256
IMG_COL=512
IMG_CHANNEL=3
#CITYSCRAPE_KIND=12
def img_mask_arr(path):
    img=cv2.imread(path)
    img1=np.zeros((IMG_ROW,int(IMG_COL/2),IMG_CHANNEL),dtype=np.float32)
    mask1=np.zeros((IMG_ROW,int(IMG_COL/2)))
    for i in range(IMG_ROW):
        for j in range(int(IMG_COL/2)):
            for k in range(IMG_CHANNEL):
                img1[i][j][k]=img[i][j][k]/255
    for i in range(IMG_ROW):
        for j in range(int(IMG_COL/2),IMG_COL):
            str1=''
            for k in range(3):
                str1+=str(img[i][j][k])
            mask1[i][j-int(IMG_COL/2)]=str1
    return img1,mask1
imglist=[]
masklist=[]
for path in os.listdir(TRAIN_IMG_DIR):
    img,mask=img_mask_arr(TRAIN_IMG_DIR+path)
    imglist.append(img)
    masklist.append(mask)
imglist=np.asarray(imglist)
masklist=np.asarray(masklist)
labels=np.unique(masklist)
print(len(labels))
def build_model(img_w,img_h,filters):
    kernel=3
    encoding_layers=[
        Conv2D(64,(kernel,kernel),input_shape=(img_w,img_h,3),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        
        Convolution2D(128,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        
        Convolution2D(256,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]
    autoencoder=Sequential()
    autoencoder.encoding_layers=encoding_layers
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    decoding_layers=[
        UpSampling2D(),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        
        UpSampling2D(),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        
        UpSampling2D(),
        Convolution2D(256,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        
        UpSampling2D(),
        Convolution2D(128,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        
        UpSampling2D(),
        Convolution2D(64,(kernel,kernel),padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(CITYSCRAPE_KIND,(1,1),padding='valid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers=decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    autoencoder.add(Reshape((CITYSCRAPE_KIND,img_h*img_w)))
    autoencoder.add(Permute((2,1)))
    autoencoder.add(Activation('softmax'))
    autoencoder.summary()
    return autoencoder
model=build_model(IMG_ROW,int(IMG_COL/2),10)