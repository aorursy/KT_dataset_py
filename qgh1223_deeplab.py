import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread,imshow
from skimage.transform import resize
import os
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
from keras.layers import Activation,Conv2D,MaxPooling2D,BatchNormalization
from keras.layers import Input,DepthwiseConv2D,add,Dropout,AveragePooling2D,Concatenate
from keras.models import Model
from keras.engine import Layer,InputSpec
from keras.utils import conv_utils
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import tensorflow as tf
from keras.optimizers import Adam
IMG_ROW=IMG_COL=64
IMG_CHANNEL=3
TRAIN_IMG_DIR='../input/train/images/'
TRAIN_MASK_DIR='../input/train/masks/'
train_img_list=os.listdir(TRAIN_IMG_DIR)
train_mask_list=os.listdir(TRAIN_MASK_DIR)
print(train_img_list[:20])
print(train_mask_list[:20])

class BilinearUpsampling(Layer):



    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):



        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)

        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')

        self.input_spec = InputSpec(ndim=4)



    def compute_output_shape(self, input_shape):

        height=self.upsampling[0]*input_shape[1] if input_shape[1] is not None else None
        width=self.upsampling[1]*input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],height,width,input_shape[3])



    def call(self, inputs):

        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),

                                               int(inputs.shape[2]*self.upsampling[1])))



    def get_config(self):

        config = {'size': self.upsampling,

              'data_format': self.data_format}

        base_config = super(BilinearUpsampling, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

def xception_downsample_block(x,channels,
                             top_relu=False):
    if(top_relu):
        x=Activation('relu')(x)
    x=DepthwiseConv2D((3,3),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=DepthwiseConv2D((3,3),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=DepthwiseConv2D((3,3),strides=(2,2),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    return x
def res_xception_downsample_block(x,channels):
    res=Conv2D(channels,(1,1),strides=(2,2),padding='same',use_bias=False)(x)
    res=BatchNormalization()(res)
    x=xception_downsample_block(x,channels)
    x=add([x,res])
    return x
def xception_block(x,channels):
    x=Activation('relu')(x)
    x=DepthwiseConv2D((3,3),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=DepthwiseConv2D((3,3),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=DepthwiseConv2D((3,3),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding='same',use_bias=False)(x)
    x=BatchNormalization()(x)
    return x
def res_xception_block(x,channels):
    res=x
    x=xception_block(x,channels)
    x=add([x,res])
    return x
def aspp(x,input_shape,out_stride):

    b0=Conv2D(256,(1,1),padding="same",use_bias=False)(x)

    b0=BatchNormalization()(b0)

    b0=Activation("relu")(b0)



    b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)

    b1=BatchNormalization()(b1)

    b1=Activation("relu")(b1)

    b1=Conv2D(256,(1,1),padding="same",use_bias=False)(b1)

    b1=BatchNormalization()(b1)

    b1=Activation("relu")(b1)



    b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)

    b2=BatchNormalization()(b2)

    b2=Activation("relu")(b2)

    b2=Conv2D(256,(1,1),padding="same",use_bias=False)(b2)

    b2=BatchNormalization()(b2)

    b2=Activation("relu")(b2)



    b3=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)

    b3=BatchNormalization()(b3)

    b3=Activation("relu")(b3)

    b3=Conv2D(256,(1,1),padding="same",use_bias=False)(b3)

    b3=BatchNormalization()(b3)

    b3=Activation("relu")(b3)



    out_shape=int(input_shape[0]/out_stride)

    b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)

    b4=Conv2D(256,(1,1),padding="same",use_bias=False)(b4)

    b4=BatchNormalization()(b4)

    b4=Activation("relu")(b4)

    b4=BilinearUpsampling((out_shape,out_shape))(b4)



    x=Concatenate()([b4,b0,b1,b2,b3])

    return x
def deeplabv3_plus(input_shape=(IMG_ROW,IMG_COL,3),out_stride=16,num_classes=21):

    img_input=Input(shape=input_shape)

    x=Conv2D(32,(3,3),strides=(2,2),padding="same",use_bias=False)(img_input)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=Conv2D(64,(3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)



    x=res_xception_downsample_block(x,128)



    res=Conv2D(256,(1,1),strides=(2,2),padding="same",use_bias=False)(x)

    res=BatchNormalization()(res)

    x=Activation("relu")(x)

    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)

    skip=BatchNormalization()(x)

    x=Activation("relu")(skip)

    x=DepthwiseConv2D((3,3),strides=(2,2),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=add([x,res])



    x=xception_downsample_block(x,728,top_relu=True)



    for i in range(16):

        x=res_xception_block(x,728)



    res=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)

    res=BatchNormalization()(res)

    x=Activation("relu")(x)

    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(728,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=add([x,res])



    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(1536,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(1536,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Conv2D(2048,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)



    #aspp

    x=aspp(x,input_shape,out_stride)

    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=Dropout(0.9)(x)



    ##decoder

    x=BilinearUpsampling((4,4))(x)

    dec_skip=Conv2D(48,(1,1),padding="same",use_bias=False)(skip)

    dec_skip=BatchNormalization()(dec_skip)

    dec_skip=Activation("relu")(dec_skip)

    x=Concatenate()([x,dec_skip])



    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)



    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)

    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)

    x=BatchNormalization()(x)

    x=Activation("relu")(x)



    x=Conv2D(num_classes,(1,1),padding="same")(x)

    x=BilinearUpsampling((4,4))(x)

    model=Model(img_input,x)

    return model
def get_img_mask_array(imgpath,maskpath):
    #print(imgpath)
    img=imread(imgpath)[:,:,:IMG_CHANNEL]
    img=resize(img,(IMG_ROW,IMG_COL),mode='constant',
               preserve_range=True)
    mask=np.zeros((IMG_ROW,IMG_COL,1),dtype=np.bool)
    mask_=imread(maskpath)
    mask_=np.expand_dims(resize(mask_,(IMG_ROW,IMG_COL),
                                mode='constant',preserve_range=True),
                         axis=-1)
    mask=np.maximum(mask,mask_)
    return np.asarray(img),np.asarray(mask)
def generate_imgarr_maskarr():
    imgarr=np.zeros((len(train_img_list),IMG_ROW,IMG_COL,IMG_CHANNEL))
    maskarr=np.zeros((len(train_img_list),IMG_ROW,IMG_COL,1),dtype=np.bool)
    for i in range(len(train_img_list)):
        img,mask=get_img_mask_array(TRAIN_IMG_DIR+train_img_list[i],TRAIN_MASK_DIR+train_img_list[i])
        imgarr[i]=img
        maskarr[i]=mask
    return imgarr,maskarr
imgarr,maskarr=generate_imgarr_maskarr()
train_imgarr,valid_imgarr,train_maskarr,valid_maskarr=train_test_split(imgarr,maskarr,
                                                                      test_size=0.1)
def train_gen(batch_size=100):
    while(True):
        imgarr=[]
        maskarr=[]
        for i in range(batch_size):
            rnd_id=random.randint(0,len(train_imgarr)-1)
            imgarr.append(train_imgarr[rnd_id])
            maskarr.append(train_maskarr[rnd_id])
        yield (np.asarray(imgarr),np.asarray(maskarr))
        imgarr=[]
        maskarr=[]
model=deeplabv3_plus(num_classes=1)
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
            score = tf.identity(score) #!! use identity to transform score to tensor
        prec.append(score) 
        
    return K.mean(K.stack(prec), axis=0)
callbacks=[
    ReduceLROnPlateau(patience=5,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('salt_model.h5',save_best_only=True,verbose=1)
]
model.compile(Adam(lr=0.01),metrics=[mean_iou],loss=[dice_coef_loss])
history=model.fit_generator(train_gen(),epochs=70,
                           steps_per_epoch=100,
                           validation_data=(valid_imgarr,valid_maskarr),
                           callbacks=callbacks)
plt.plot(history.history['mean_iou'])
plt.plot(history.history['val_mean_iou'])
plt.legend(['train','valid'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])
predictresult=model.predict(train_imgarr)
print(predictresult.shape)
num=331
for i in range(3):
    rnd_id=random.randint(0,len(predictresult)-1)
    plt.subplot(str(331+3*i))
    plt.imshow(train_imgarr[rnd_id])
    plt.subplot(str(num+3*i+1))
    plt.imshow(np.squeeze(train_maskarr[rnd_id]))
    plt.subplot(str(331+3*i+2))
    plt.imshow(np.squeeze(predictresult[rnd_id]))