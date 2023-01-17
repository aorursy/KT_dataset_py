
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cv2
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from keras.models import Model,load_model
from keras.layers import Input,Dropout,Activation,UpSampling2D
from keras.layers.core import Lambda,RepeatVector,Reshape,SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
shipdata=pd.read_csv('../input/train_ship_segmentations.csv')
print(shipdata.head())
print(len(shipdata))
train_dir='../input/train/'
test_dir='../input/test/'
IMG_RAW_COL=IMG_RAW_ROW=768
IMG_ROW=IMG_COL=128
IMG_CHANNEL=3
train_imgpath,valid_imgpath,train_maskstr,valid_maskstr=train_test_split(shipdata['ImageId'],shipdata['EncodedPixels'],
                                                                        test_size=0.08)
def imgarr(imgpath):
    img=cv2.imread(imgpath)
    img=cv2.resize(img,(IMG_RAW_COL,IMG_RAW_ROW))
    return img
def mask_decode(maskstr):
    img=np.zeros(IMG_RAW_COL*IMG_RAW_ROW,dtype=np.uint8)
    if(type(maskstr)==np.float):
        return img.reshape((IMG_RAW_COL,IMG_RAW_ROW)).T
    s=maskstr.split(' ')
    starts,lengths=[np.asarray(x,dtype=np.int32) for x in (s[0:][::2],s[1:][::2])]
    starts-=1
    ends=starts+lengths
    for lo,hi in zip(starts,ends):
        img[lo:hi]=1
    return img.reshape((IMG_RAW_COL,IMG_RAW_ROW)).T
for i in range(25):
    rnd_id=random.randint(0,len(shipdata)-1)
    f,ax=plt.subplots(1,2,figsize=(15,2))
    axes=ax.flatten()
    j=0
    for ax in axes:
        imgpath=train_dir+shipdata['ImageId'][rnd_id]
        imgarray=imgarr(imgpath)
        maskarray=mask_decode(shipdata['EncodedPixels'][rnd_id])
        if(j==0):
            ax.imshow(imgarray)
        else:
            ax.imshow(maskarray)
        j+=1
def transform_imgarr(imgpath):
    img=cv2.imread(imgpath)
    img=cv2.resize(img,(IMG_COL,IMG_ROW))
    return img
def transform_num(num):
    return int(int(num)*(IMG_COL**2)/(IMG_RAW_COL**2))
def transform_maskarr(maskstr):
    mask=np.zeros(IMG_COL*IMG_ROW,dtype=np.bool)
    if(len(maskstr)==0):
        return mask.reshape((IMG_ROW,IMG_COL,1))
    s=maskstr.split(' ')
    for i in range(len(s)):
        s[i]=transform_num(s[i])
    starts,lengths=[np.asarray(x,dtype=np.int32) for x in (s[0:][::2],s[1:][::2])]
    starts-=1
    ends=starts+lengths
    for lo,hi in zip(starts,ends):
        mask[lo:hi]=1
    mask=mask.reshape((IMG_COL,IMG_ROW,1))
    return mask
def train_gen(batch_size=200):
    imgarr=[]
    maskarr=[]
    while(True):
        for i in range(batch_size):
            rnd_id=random.randint(0,len(train_imgpath)-1)
            img=transform_imgarr(train_dir+train_imgpath[train_imgpath.index[rnd_id]])
            if(type(train_maskstr[train_imgpath.index[rnd_id]])==np.float):
                mask=transform_maskarr('')
            else:
                mask=transform_maskarr(train_maskstr[train_maskstr.index[rnd_id]])
            imgarr.append(img)
            maskarr.append(mask)
        yield (np.asarray(imgarr),np.asarray(maskarr))
        imgarr=[]
        maskarr=[]
def testdata():
    imgarr=[]
    maskarr=[]
    for i in range(len(valid_imgpath)):
        img=transform_imgarr(train_dir+train_imgpath[train_imgpath.index[i]])
        if(type(train_maskstr[train_maskstr.index[i]])==np.float):
            mask=transform_maskarr('')
        else:
            mask=transform_maskarr(train_maskstr[train_maskstr.index[i]])
        imgarr.append(img)
        maskarr.append(mask)
    return (np.asarray(imgarr),np.asarray(maskarr))
(imgarr,maskarr)=testdata()
print(imgarr.shape)
print(maskarr.shape)
def double_conv_layer(x,size,dropout=0.0,batch_norm=True):
    conv=Conv2D(size,(3,3),padding='same')(x)
    if(batch_norm):
        conv=BatchNormalization(axis=3)(conv)
    conv=Activation('sigmoid')(conv)
    conv=Conv2D(size,(3,3),padding='same')(conv)
    if(batch_norm):
        conv=BatchNormalization(axis=3)(conv)
    conv=Activation('relu')(conv)
    if(dropout!=0.0):
        conv=SpatialDropout2D(dropout)(conv)
    return conv

def build_unet_model(filters):
    inputs=Input((IMG_ROW,IMG_COL,IMG_CHANNEL))
    conv1=double_conv_layer(inputs,filters)
    p1=MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2=double_conv_layer(p1,2*filters)
    p2=MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3=double_conv_layer(p2,4*filters)
    p3=MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4=double_conv_layer(p3,8*filters)
    p4=MaxPooling2D(pool_size=(2,2))(conv4)
    
    conv5=double_conv_layer(p4,16*filters)
    p5=MaxPooling2D(pool_size=(2,2))(conv5)
    
    conv6=double_conv_layer(p5,32*filters)
    
    up7=concatenate([UpSampling2D(size=(2,2))(conv6),conv5],axis=3)
    conv7=double_conv_layer(up7,16*filters)
    
    up8=concatenate([UpSampling2D(size=(2,2))(conv7),conv4],axis=3)
    conv8=double_conv_layer(up8,8*filters)
    
    up9=concatenate([UpSampling2D(size=(2,2))(conv8),conv3],axis=3)
    conv9=double_conv_layer(up9,4*filters)
    
    up10=concatenate([UpSampling2D(size=(2,2))(conv9),conv2],axis=3)
    conv10=double_conv_layer(up10,2*filters)
    
    up11=concatenate([UpSampling2D(size=(2,2))(conv10),conv1],axis=3)
    conv11=double_conv_layer(up11,filters,0)
    
    convfinal=Conv2D(1,(1,1))(conv11)
    convfinal=Activation('sigmoid')(convfinal)
    
    model=Model(inputs,convfinal)
    model.summary()
    return model

model=build_unet_model(8)
def dice_coef(y_true,y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return (2.0*intersection+1.0)/(K.sum(y_true_f)+K.sum(y_pred_f)+1.0)
def dice_coef_loss(y_true,y_pred):
    return -dice_coef(y_true,y_pred)
def mean_iou(Y_true, Y_pred, score_thres=0.5):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        Y_pred_bool = tf.to_int32(Y_pred > t) # boolean mask by threshold
        score, update_op = tf.metrics.mean_iou(Y_true, Y_pred_bool, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            score = tf.identity(score)
        prec.append(score) 
    return K.mean(K.stack(prec), axis=0)
callbacks=[
    ReduceLROnPlateau(patience=5,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('air_model.h5',save_best_only=True,verbose=1)
]
model.compile(Adam(lr=0.001),metrics=[mean_iou],loss='binary_crossentropy')
history=model.fit_generator(train_gen(),epochs=50,
                           steps_per_epoch=100,
                           validation_data=(imgarr,maskarr),
                           callbacks=callbacks)
plt.plot(history.history['mean_iou'])
plt.plot(history.history['val_mean_iou'])