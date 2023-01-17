import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from skimage.io import imread,imshow
from skimage.transform import resize
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import *
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
IMG_ROW=IMG_COL=64
IMG_CHANNEL=3
TRAIN_IMG_DIR='../input/train/images/'
TRAIN_MASK_DIR='../input/train/masks/'
TEST_IMG_DIR='../input/test/images/'
imgpathlist=os.listdir(TRAIN_IMG_DIR)
maskpathlist=os.listdir(TRAIN_MASK_DIR)
imgarray=np.zeros((len(imgpathlist),IMG_ROW,IMG_COL,IMG_CHANNEL),dtype=np.uint8)
maskarray=np.zeros((len(maskpathlist),IMG_ROW,IMG_COL,1),dtype=np.bool)
for i in range(len(imgpathlist)):
    imgpath=TRAIN_IMG_DIR+imgpathlist[i]
    maskpath=TRAIN_MASK_DIR+maskpathlist[i]
    img=imread(imgpath)[:,:,:IMG_CHANNEL]
    img=resize(img,(IMG_ROW,IMG_COL),mode='constant',
               preserve_range=True)
    imgarray[i]=img
    mask=np.zeros((IMG_ROW,IMG_COL,1),
                  dtype=np.bool)
    mask_=imread(maskpath)
    mask_=np.expand_dims(resize(mask_,(IMG_ROW,IMG_COL),
                               mode='constant',preserve_range=True),
                        axis=-1)
    mask=np.maximum(mask,mask_)
    maskarray[i]=mask
print(imgarray.shape)
print(maskarray.shape)
for i in range(10):
    rnd_id=random.randint(0,len(imgarray)-1)
    f,ax=plt.subplots(1,2,figsize=(15,2))
    axes=ax.flatten()
    j=0
    for ax in axes:
        if(j==0):
            ax.imshow(imgarray[rnd_id],
                     cmap='seismic')
        else:
            ax.imshow(np.squeeze(maskarray[rnd_id]),
                     cmap='seismic')
        j+=1
img_train,img_test,mask_train,mask_test=train_test_split(imgarray,maskarray,
                                                        test_size=0.15)
print(img_train.shape)
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
def double_conv_layer(x,size,dropout=0.0,batch_norm=True):
    conv=Conv2D(size,(3,3),padding='same')(x)
    if(batch_norm==True):
        conv=BatchNormalization(axis=3)(conv)
    conv=Activation('relu')(conv)
    conv=Conv2D(size,(3,3),padding='same')(conv)
    if(batch_norm==True):
        conv=BatchNormalization(axis=3)(conv)
    conv=Activation('relu')(conv)
    return conv
def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation('relu')(x)
    return x

def double_residual_layer(blockInput, num_filters=16):
    blockInput=Conv2D(num_filters,(1,1),padding='same')(blockInput)
    x = Activation('relu')(blockInput)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x
def build_zf_unet(filters):
    inputs=Input((IMG_ROW,IMG_COL,IMG_CHANNEL))
    conv1=double_residual_layer(inputs,filters)
    p1=MaxPooling2D(pool_size=(2,2))(conv1)

    conv2=double_residual_layer(p1,2*filters)
    p2=MaxPooling2D(pool_size=(2,2))(conv2)

    conv3=double_residual_layer(p2,4*filters)
    p3=MaxPooling2D(pool_size=(2,2))(conv3)

    conv4=double_residual_layer(p3,8*filters)
    p4=MaxPooling2D(pool_size=(2,2))(conv4)

    conv5=double_residual_layer(p4,16*filters)
    p5=MaxPooling2D(pool_size=(2,2))(conv5)

    conv6=double_residual_layer(p5,32*filters)

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
model=build_zf_unet(8)
callbacks=[
    #EarlyStopping(patience=5,monitor='val_loss',verbose=1),
    ReduceLROnPlateau(patience=3,monitor='val_loss',verbose=1),
    ModelCheckpoint('model.h5',verbose=1,save_best_only=True)
]
results=model.fit(img_train,mask_train,callbacks=callbacks,epochs=50,
                 validation_data=(img_test,mask_test))
plt.plot(results.history['mean_iou'])
plt.plot(results.history['val_mean_iou'])
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
model.load_weights('model.h5')
predict_maskarray=model.predict(imgarray)
for i in range(10):
    rnd_id=random.randint(0,len(imgarray)-1)
    f,ax=plt.subplots(1,3,figsize=(15,2))
    axes=ax.flatten()
    j=0
    for ax in axes:
        if(j==0):
            ax.imshow(imgarray[rnd_id])
        elif(j==1):
            ax.imshow(np.squeeze(maskarray[rnd_id]))
        else:
            ax.imshow(np.squeeze(predict_maskarray[rnd_id]))
        j+=1
testimglist=os.listdir(TEST_IMG_DIR)
test_imgarray=np.zeros((len(testimglist),
                        IMG_ROW,IMG_COL,IMG_CHANNEL),
                      dtype=np.float32)
for i in range(len(testimglist)):
    IMGPATH=TEST_IMG_DIR+testimglist[i]
    img=imread(IMGPATH)[:,:,:IMG_CHANNEL]
    img=resize(img,(IMG_ROW,IMG_COL),mode='constant',
               preserve_range=True)
    test_imgarray[i]=img
test_maskarray=model.predict(test_imgarray)
for i in range(10):
    rnd_id=random.randint(0,len(test_imgarray)-1)
    f,ax=plt.subplots(1,2,figsize=(15,2))
    axes=ax.flatten()
    j=0
    for ax in axes:
        if(j==0):
            ax.imshow(test_imgarray[rnd_id],cmap='seismic')
        else:
            ax.imshow(np.squeeze(test_maskarray[rnd_id]),cmap='seismic')
        j+=1
pred_test_result=[]
for i in range(len(test_maskarray)):
    pred_test_result.append(resize(np.squeeze(test_maskarray),
                                   (101,101),
                                  mode='constant',
                                  preserve_range=True))
print(pred_test_result.shape)