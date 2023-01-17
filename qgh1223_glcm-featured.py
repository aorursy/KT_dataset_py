import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from skimage.io import imread
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import resize
from multiprocessing import Pool
import random
import matplotlib.pyplot as plt
%matplotlib inline
import tqdm
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import Model,Sequential
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf

IMG_ROW=IMG_COL=32
IMG_CHANNEL=3
TRAIN_IMG_DIR='../input/train/images/'
TRAIN_MASK_DIR='../input/train/masks/'
imgidlist=os.listdir(TRAIN_IMG_DIR)
maskidlist=os.listdir(TRAIN_MASK_DIR)
train_img_list,valid_img_list,train_mask_list,valid_mask_list=train_test_split(imgidlist,maskidlist,
                                                                             test_size=0.01)
def glcm_props(patch):
    lf = []
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']

    # left nearest neighbor
    glcm = greycomatrix(patch, [1], [0], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( greycoprops(glcm, f)[0,0] )

    # upper nearest neighbor
    glcm = greycomatrix(patch, [1], [np.pi/2], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( greycoprops(glcm, f)[0,0] )
        
    return lf

def patch_gen(img, PAD=4):
    img1 = (img * 255).astype(np.uint8)

    W = 101
    imgx = np.zeros((101+PAD*2, 101+PAD*2), dtype=img1.dtype)
    imgx[PAD:W+PAD,PAD:W+PAD] = img1
    imgx[:PAD,  PAD:W+PAD] = img1[PAD:0:-1,:]
    imgx[-PAD:, PAD:W+PAD] = img1[W-1:-PAD-1:-1,:]
    imgx[:, :PAD ] = imgx[:, PAD*2:PAD:-1]
    imgx[:, -PAD:] = imgx[:, W+PAD-1:-PAD*2-1:-1]

    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, W))
    xx, yy = xx.flatten() + PAD, yy.flatten() + PAD

    for x, y in zip(xx, yy):
        patch = imgx[y-PAD:y+PAD+1, x-PAD:x+PAD+1]
        yield patch

def glcm_feature(img, verbose=False):
    
    W, NF, PAD = 101, 10, 4

    if img.sum() == 0:
        return np.zeros((W,W,NF), dtype=np.float32)
    
    l = []
    with Pool(3) as pool:
        for p in tqdm.tqdm(pool.imap(glcm_props, patch_gen(img, PAD)), total=W*W, disable=not verbose):
            l.append(p)
        
    fimg = np.array(l, dtype=np.float32).reshape(101, 101, -1)
    
    return fimg

def read_image(imgpath):
    img=imread(imgpath).astype(np.uint8)
    img1=imread(imgpath)[...,0].astype(np.float32) / 255
    return img,img1
def process_img(imgpath):
    img,img_=read_image(imgpath)
    fimg=glcm_feature(img_,verbose=0)
    amin=np.amin(fimg,axis=(0,1))
    amax=np.amax(fimg,axis=(0,1))
    fimg=(fimg-amin)/(amax-amin)
    img1=np.power(fimg[...,4],3)
    img2=np.power(fimg[...,9],3)
    img1=resize(img1,(IMG_ROW,IMG_COL))
    img2=resize(img2,(IMG_ROW,IMG_COL))
    img1=transform_img(img1)
    img2=transform_img(img2)
    return img1,img2
def read_mask(maskpath):
    mask=np.zeros((IMG_ROW,IMG_COL,1),
                  dtype=np.bool)
    mask_=imread(maskpath)
    mask_=np.expand_dims(resize(mask_,(IMG_ROW,IMG_COL),
                               mode='constant',preserve_range=True),
                        axis=-1)
    mask=np.maximum(mask,mask_)
def transform_img(img):
    img1=np.zeros((IMG_ROW,IMG_COL,IMG_CHANNEL))
    for i in range(IMG_ROW):
        for j in range(IMG_COL):
            for k in range(IMG_CHANNEL):
                img1[i][j][k]=img[i][j]
    return img1
def visual_glcm():
    for i in range(10):
        rndid=random.randint(0,len(imgidlist)-1)
        img,img1=read_image(TRAIN_IMG_DIR+imgidlist[rndid])
        mask,mask1=read_image(TRAIN_MASK_DIR+imgidlist[rndid])
        _,(ax0,ax1,ax2,ax3)=plt.subplots(1,4,figsize=(6,2))
        ax0.imshow(img)
        ax1.imshow(mask)
        fimg = glcm_feature(img1, verbose=0)
        amin = np.amin(fimg, axis=(0,1))
        amax = np.amax(fimg, axis=(0,1))
        fimg = (fimg - amin) / (amax - amin)
        fimg[...,4] = np.power(fimg[...,4], 3)
        fimg[...,9] = np.power(fimg[...,9], 3)
        img1=fimg[...,4]
        img2=fimg[...,9]
        img1=resize(img1,(IMG_ROW,IMG_COL))
        img2=resize(img2,(IMG_ROW,IMG_COL))
        img1=transform_img(img1)
        img2=transform_img(img2)
        print(img1.shape)
        ax2.imshow(img1,cmap='seismic')
        ax3.imshow(img2,cmap='seismic')
def train_gen(batch_size=128):
    while(True):
        imglist=np.zeros((batch_size,IMG_ROW,IMG_COL,IMG_CHANNEL))
        masklist=np.zeros((batch_size,IMG_ROW,IMG_COL,1),dtype=np.bool)
        for i in range(batch_size):
            rnd_id=random.randint(0,len(train_img_list)-1)
            imgpath=TRAIN_IMG_DIR+train_img_list[rnd_id]
            maskpath=TRAIN_MASK_DIR+train_mask_list[rnd_id]
            img1,img2=process_img(imgpath)
            mask=read_mask(maskpath)
            imglist[i]=img1
            masklist[i]=mask
        yield (imglist,masklist)
        imglist=np.zeros((batch_size,IMG_ROW,IMG_COL,IMG_CHANNEL))
        masklist=np.zeros((batch_size,IMG_ROW,IMG_COL,1),dtype=np.bool)
def valid_gen():
    imglist=np.zeros((len(valid_img_list),IMG_ROW,IMG_COL,IMG_CHANNEL))
    masklist=np.zeros((len(valid_img_list),IMG_ROW,IMG_COL,1),dtype=np.bool)
    for i in range(len(valid_img_list)):
        imgpath=TRAIN_IMG_DIR+valid_img_list[i]
        maskpath=TRAIN_MASK_DIR+valid_mask_list[i]
        img1,img2=process_img(imgpath)
        mask=read_mask(maskpath)
        imglist[i]=img1
        masklist[i]=mask
    return (imglist,masklist)
validdata=valid_gen()
(validimglist,validmasklist)=validdata
print(validimglist.shape)
print(validmasklist.shape)
visual_glcm()
def double_conv_layer(x,size,dropout=0.0,batch_norm=True):
    conv=Conv2D(size,(2,2),padding='same')(x)
    if(batch_norm==True):
        conv=BatchNormalization(axis=3)(conv)
    conv=Activation('relu')(conv)
    conv=Conv2D(size,(3,3),padding='same')(conv)
    if(batch_norm==True):
        conv=BatchNormalization(axis=3)(conv)
    conv=Activation('relu')(conv)
    if(dropout!=0.0):
        conv=SpatialDropout(dropout)(conv)
    return conv
def unet_model(filters):
    inputs=Input((IMG_ROW,IMG_COL,IMG_CHANNEL))
    conv1=double_conv_layer(inputs,filters)
    p1=MaxPooling2D(pool_size=(2,2))(conv1)

    conv2=double_conv_layer(p1,2*filters)
    p2=MaxPooling2D(pool_size=(2,2))(conv2)

    conv3=double_conv_layer(p2,4*filters)
    p3=MaxPooling2D(pool_size=(2,2))(conv3)

    conv4=double_conv_layer(p3,8*filters)
    p4=MaxPooling2D(pool_size=(2,2))(conv4)

    conv5=double_conv_layer(p4,32*filters)

    up6=concatenate([UpSampling2D(size=(2,2))(conv5),conv4],axis=3)
    conv6=double_conv_layer(up6,8*filters)

    up7=concatenate([UpSampling2D(size=(2,2))(conv6),conv3],axis=3)
    conv7=double_conv_layer(up7,4*filters)

    up8=concatenate([UpSampling2D(size=(2,2))(conv7),conv2],axis=3)
    conv8=double_conv_layer(up8,2*filters)

    up9=concatenate([UpSampling2D(size=(2,2))(conv8),conv1],axis=3)
    conv9=double_conv_layer(up9,filters,0)

    convfinal=Conv2D(1,(1,1))(conv9)
    convfinal=Activation('sigmoid')(convfinal)

    model=Model(inputs,convfinal)
    model.summary()
    return model
model=unet_model(8)

def dice_coef(y_true,y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return (2.0*intersection+1.0)/(K.sum(y_true_f)+K.sum(y_pred_f)+1.0)
def dice_coef_loss(y_true,y_pred):
    return -dice_coef(y_true,y_pred)
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
model.compile(Adam(lr=0.001),metrics=[mean_iou],loss=dice_coef_loss)
callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=3,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('salt_model.h5',monitor='val_loss',save_best_only=True,verbose=1)
]