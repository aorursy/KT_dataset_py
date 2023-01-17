!pip install easydict

!git clone https://github.com/onnx/models.git



import os

import sys

sys.path.insert(0, "models/vision/body_analysis/arcface/")
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import cv2

import numpy as np

import pandas as pd

import mxnet as mx

import keras

import random

import sklearn

from scipy import misc

import matplotlib.pyplot as plt



from sklearn.decomposition import PCA

from time import sleep

from easydict import EasyDict as edict

from skimage import transform as trans

from mtcnn_detector import MtcnnDetector



from mxnet.contrib.onnx.onnx2mx.import_model import import_model

import tqdm

from keras.models import Sequential, Model

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, PReLU, Add, Dropout, Permute

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense

from keras.engine.topology import Layer

from keras import backend as K

import tensorflow as tf
def get_model(ctx, model):

    image_size = (112,112)

    # Import ONNX model

    sym, arg_params, aux_params = import_model(model)

    # Define and binds parameters to the network

    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)

    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])

    model.set_params(arg_params, aux_params)

    return model



def preprocess(img, bbox=None, landmark=None, **kwargs):

    M = None

    image_size = []

    str_image_size = kwargs.get('image_size', '')

    # Assert input shape

    if len(str_image_size)>0:

        image_size = [int(x) for x in str_image_size.split(',')]

        if len(image_size)==1:

            image_size = [image_size[0], image_size[0]]

        assert len(image_size)==2

        #assert image_size[0]==112

        assert image_size[0]==112 or image_size[1]==96

    

    # Do alignment using landmark points

    if landmark is not None:

        assert len(image_size)==2

        src = np.array([

          [30.2946, 51.6963],

          [65.5318, 51.5014],

          [48.0252, 71.7366],

          [33.5493, 92.3655],

          [62.7299, 92.2041] ], dtype=np.float32 )

        if image_size[1]==112:

            src[:,0] += 8.0

        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()

        tform.estimate(dst, src)

        M = tform.params[0:2,:]

        assert len(image_size)==2

        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

        ret = cv2.resize(warped, (96, 96))

        return ret

    

    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop

    if M is None:

        if bbox is None:

            det = np.zeros(4, dtype=np.int32)

            det[0] = int(img.shape[1]*0.0625)

            det[1] = int(img.shape[0]*0.0625)

            det[2] = img.shape[1] - det[0]

            det[3] = img.shape[0] - det[1]

        else:

            det = bbox

        margin = kwargs.get('margin', 44)

        bb = np.zeros(4, dtype=np.int32)

        bb[0] = np.maximum(det[0]-margin/2, 0)

        bb[1] = np.maximum(det[1]-margin/2, 0)

        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])

        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])

        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]

        if len(image_size)>0:

            ret = cv2.resize(ret, (96, 96))

            

        return ret

    

def get_input(detector,face_img):

    # Pass input images through face detector

    ret = detector.detect_face(face_img, det_type = 0)

    if ret is None:

        nimg=preprocess(face_img, bbox=None, points=None, image_size='112,112')

        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

        aligned = np.transpose(nimg, (2,0,1))

        return aligned

    bbox, points = ret

    if bbox.shape[0]==0:

        return None

    bbox = bbox[0,0:4]

    points = points[0,:].reshape((2,5)).T

    # Call preprocess() to generate aligned images

    #nimg = cv2.resize(preprocess(face_img, bbox, points, image_size='112,112'),(96,96))

    nimg=preprocess(face_img, bbox, points, image_size='112,112')

    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)

    aligned = np.transpose(nimg, (2,0,1))

    return aligned

def Createheadmodel():

    inp=keras.layers.Input((128,))

    x=keras.layers.Dense(64,activation='elu')(inp)

    x=keras.layers.Dropout(0.1)(x)

    out=keras.layers.Dense(1,activation='hard_sigmoid',use_bias=False,kernel_initializer=keras.initializers.Ones())(x)

    model=keras.models.Model(input=inp,output=out)

    model.layers[-1].trainable=False

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse')

    return model
for i in range(4):

    mx.test_utils.download(dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-0001.params'.format(i+1))

    mx.test_utils.download(dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-symbol.json'.format(i+1))

    mx.test_utils.download(dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.caffemodel'.format(i+1))

    mx.test_utils.download(dirname='mtcnn-model', url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.prototxt'.format(i+1))

# Determine and set context

if len(mx.test_utils.list_gpus())==0:

    ctx = mx.cpu()

else:

    ctx = mx.gpu(0)

# Configure face detector

det_threshold = [0.6,0.7,0.8]

mtcnn_path = os.path.join(os.path.dirname('__file__'), 'mtcnn-model')

detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=det_threshold)
ratingDS=pd.read_excel('../input/faces-scut/scut-fbp5500_v2/SCUT-FBP5500_v2/All_Ratings.xlsx')

Answer=ratingDS.groupby('Filename').mean()['Rating']
ratingDS['race']=ratingDS['Filename'].apply(lambda x:x[:2])

plt.rcParams["figure.figsize"] = [16,9]

fig, ax = plt.subplots(2, 2, sharex='col')



for i, race in enumerate(['CF','CM','AF','AM']):

    sbp=ax[i%2,i//2]

    ratingDS[ratingDS['race']==race].groupby('Filename')['Rating'].mean().hist(alpha=0.5, bins=20,label=race,grid=False,rwidth=0.9,ax=sbp)

    sbp.set_title(race)
ratingDS.groupby('Filename')['Rating'].std().mean()
R2=ratingDS.join(ratingDS.groupby('Filename')['Rating'].median(), on='Filename', how='inner',rsuffix =' median')

R2['ratingdiff']=(R2['Rating median']-R2['Rating']).astype(int)
R2['ratingdiff'].hist(label='difference of raitings',bins=[-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5,4.5],grid=False,rwidth=0.5)
len(R2[R2['ratingdiff'].abs()>1])/len(R2)
def conv2d_bn(

  x,

  layer=None,

  cv1_out=None,

  cv1_filter=(1, 1),

  cv1_strides=(1, 1),

  cv2_out=None,

  cv2_filter=(3, 3),

  cv2_strides=(1, 1),

  padding=None,

):

    num = '' if cv2_out == None else '1'

    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, name=layer+'_conv'+num)(x)

    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer+'_bn'+num)(tensor)

    tensor = Activation('relu')(tensor)

    if padding == None:

        return tensor

    tensor = ZeroPadding2D(padding=padding)(tensor)

    if cv2_out == None:

        return tensor

    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, name=layer+'_conv'+'2')(tensor)

    tensor = BatchNormalization(axis=3, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)

    tensor = Activation('relu')(tensor)

    return tensor



myInput = Input(shape=(96, 96, 3))



x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)

x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)

x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)

x = Activation('relu')(x)

x = ZeroPadding2D(padding=(1, 1))(x)

x = MaxPooling2D(pool_size=3, strides=2)(x)

#x = Lambda(LRN2D, name='lrn_1')(x)

x = Conv2D(64, (1, 1), name='conv2')(x)

x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)

x = Activation('relu')(x)

x = ZeroPadding2D(padding=(1, 1))(x)

x = Conv2D(192, (3, 3), name='conv3')(x)

x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)

x = Activation('relu')(x)

#x = Lambda(LRN2D, name='lrn_2')(x)

x = ZeroPadding2D(padding=(1, 1))(x)

x = MaxPooling2D(pool_size=3, strides=2)(x)



# Inception3a

inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)

inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)

inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)

inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)

inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)

inception_3a_3x3 = Activation('relu')(inception_3a_3x3)



inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)

inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)

inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)

inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)

inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)

inception_3a_5x5 = Activation('relu')(inception_3a_5x5)



inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)

inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)

inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)

inception_3a_pool = Activation('relu')(inception_3a_pool)

inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)



inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)

inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)

inception_3a_1x1 = Activation('relu')(inception_3a_1x1)



inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)



# Inception3b

inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)

inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)

inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)

inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)

inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)

inception_3b_3x3 = Activation('relu')(inception_3b_3x3)



inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)

inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)

inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)

inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)

inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)

inception_3b_5x5 = Activation('relu')(inception_3b_5x5)



inception_3b_pool = Lambda(lambda x: x**2, name='power2_3b')(inception_3a)

inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)

inception_3b_pool = Lambda(lambda x: x*9, name='mult9_3b')(inception_3b_pool)

inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)

inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)

inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)

inception_3b_pool = Activation('relu')(inception_3b_pool)

inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)



inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)

inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)

inception_3b_1x1 = Activation('relu')(inception_3b_1x1)



inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)



# Inception3c

inception_3c_3x3 = conv2d_bn(inception_3b,

                                   layer='inception_3c_3x3',

                                   cv1_out=128,

                                   cv1_filter=(1, 1),

                                   cv2_out=256,

                                   cv2_filter=(3, 3),

                                   cv2_strides=(2, 2),

                                   padding=(1, 1))



inception_3c_5x5 = conv2d_bn(inception_3b,

                                   layer='inception_3c_5x5',

                                   cv1_out=32,

                                   cv1_filter=(1, 1),

                                   cv2_out=64,

                                   cv2_filter=(5, 5),

                                   cv2_strides=(2, 2),

                                   padding=(2, 2))



inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)

inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)



inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)



#inception 4a

inception_4a_3x3 = conv2d_bn(inception_3c,

                                   layer='inception_4a_3x3',

                                   cv1_out=96,

                                   cv1_filter=(1, 1),

                                   cv2_out=192,

                                   cv2_filter=(3, 3),

                                   cv2_strides=(1, 1),

                                   padding=(1, 1))

inception_4a_5x5 = conv2d_bn(inception_3c,

                                   layer='inception_4a_5x5',

                                   cv1_out=32,

                                   cv1_filter=(1, 1),

                                   cv2_out=64,

                                   cv2_filter=(5, 5),

                                   cv2_strides=(1, 1),

                                   padding=(2, 2))



inception_4a_pool = Lambda(lambda x: x**2, name='power2_4a')(inception_3c)

inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)

inception_4a_pool = Lambda(lambda x: x*9, name='mult9_4a')(inception_4a_pool)

inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)

inception_4a_pool = conv2d_bn(inception_4a_pool,

                                   layer='inception_4a_pool',

                                   cv1_out=128,

                                   cv1_filter=(1, 1),

                                   padding=(2, 2))

inception_4a_1x1 = conv2d_bn(inception_3c,

                                   layer='inception_4a_1x1',

                                   cv1_out=256,

                                   cv1_filter=(1, 1))

inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)



#inception4e

inception_4e_3x3 = conv2d_bn(inception_4a,

                                   layer='inception_4e_3x3',

                                   cv1_out=160,

                                   cv1_filter=(1, 1),

                                   cv2_out=256,

                                   cv2_filter=(3, 3),

                                   cv2_strides=(2, 2),

                                   padding=(1, 1))

inception_4e_5x5 = conv2d_bn(inception_4a,

                                   layer='inception_4e_5x5',

                                   cv1_out=64,

                                   cv1_filter=(1, 1),

                                   cv2_out=128,

                                   cv2_filter=(5, 5),

                                   cv2_strides=(2, 2),

                                   padding=(2, 2))

inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)

inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)



inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)



#inception5a

inception_5a_3x3 = conv2d_bn(inception_4e,

                                   layer='inception_5a_3x3',

                                   cv1_out=96,

                                   cv1_filter=(1, 1),

                                   cv2_out=384,

                                   cv2_filter=(3, 3),

                                   cv2_strides=(1, 1),

                                   padding=(1, 1))



inception_5a_pool = Lambda(lambda x: x**2, name='power2_5a')(inception_4e)

inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)

inception_5a_pool = Lambda(lambda x: x*9, name='mult9_5a')(inception_5a_pool)

inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)

inception_5a_pool = conv2d_bn(inception_5a_pool,

                                   layer='inception_5a_pool',

                                   cv1_out=96,

                                   cv1_filter=(1, 1),

                                   padding=(1, 1))

inception_5a_1x1 = conv2d_bn(inception_4e,

                                   layer='inception_5a_1x1',

                                   cv1_out=256,

                                   cv1_filter=(1, 1))



inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)



#inception_5b

inception_5b_3x3 = conv2d_bn(inception_5a,

                                   layer='inception_5b_3x3',

                                   cv1_out=96,

                                   cv1_filter=(1, 1),

                                   cv2_out=384,

                                   cv2_filter=(3, 3),

                                   cv2_strides=(1, 1),

                                   padding=(1, 1))

inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)

inception_5b_pool = conv2d_bn(inception_5b_pool,

                                   layer='inception_5b_pool',

                                   cv1_out=96,

                                   cv1_filter=(1, 1))

inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)



inception_5b_1x1 = conv2d_bn(inception_5a,

                                   layer='inception_5b_1x1',

                                   cv1_out=256,

                                   cv1_filter=(1, 1))

inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)



av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)

reshape_layer = Flatten()(av_pool)

dense_layer = Dense(128, name='dense_layer')(reshape_layer)

norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)





# Final Model

model = Model(inputs=[myInput], outputs=norm_layer)
#model.summary()


model.load_weights('../input/insightface-weights/smallfacesnn.h5')
imgpath='../input/faces-scut/scut-fbp5500_v2/SCUT-FBP5500_v2/Images/'

facevecs=[]

for name in tqdm.tqdm(Answer.index):

    img1 = cv2.imread(imgpath+name)

    pre1 = np.moveaxis(get_input(detector,img1),0,-1)

    out1 = model.predict(np.stack([pre1])/255.0)

    facevecs.append(out1)
modelhead=Createheadmodel()
X=np.stack(facevecs)[:,0,:]

Y=(Answer[:])/5

Indicies=np.arange(len(Answer))

X,Y,Indicies=sklearn.utils.shuffle(X,Y,Indicies)

Xtrain=X[:int(len(facevecs)*0.9)]

Ytrain=Y[:int(len(facevecs)*0.9)]

Indtrain=Indicies[:int(len(facevecs)*0.9)]

Xval=X[int(len(facevecs)*0.9):]

Yval=Y[int(len(facevecs)*0.9):]

Indval=Indicies[int(len(facevecs)*0.9):]
hist=modelhead.fit(Xtrain,Ytrain,

    epochs=4000,

    batch_size=5000,

    validation_data=(Xval,Yval),

    verbose=1

    )
plt.plot(hist.history['loss'][100:], label='loss')

plt.plot(hist.history['val_loss'][100:],label='validation_loss')

plt.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right', borderaxespad=0.)
hist.history['val_loss'][-1]
Answer2=Answer.to_frame()[:5500]

Answer2['ans']=0

Answer2['race']=Answer2.index

Answer2['race']=Answer2['race'].apply(lambda x: x[:2])

Answer2['ans']=modelhead.predict(np.stack(facevecs)[:,0,:])*5

xy=np.array(Answer2.iloc[Indval][['ans','Rating']])

plt.scatter(xy[:,1],xy[:,0])
import matplotlib.image as mpimg

f, axarr = plt.subplots(4,5,figsize=(10, 10))

for i, race in enumerate(['AF','CF', "AM", 'CM']):

    for rating in range(1,6):

        #axarr[i,rating-1].axis('off')

        axarr[i,rating-1].tick_params(# changes apply to the x-axis

        which='both',      # both major and minor ticks are affected

        bottom=False,      # ticks along the bottom edge are off

        top=False,         # ticks along the top edge are off

        right=False,

        left=False,

        labelbottom=False,

        labelleft=False

        )

        picname=(Answer2[Answer2['race']==race]['ans']-rating).abs().argmin()

        axarr[i,rating-1].set_xlabel(Answer2.loc[picname]['ans'])

        axarr[i,rating-1].imshow(mpimg.imread(imgpath+picname))
from IPython.display import FileLink

finmodel=Model(input=model.input, output=modelhead(model.output))

finmodel.save('finmodel.h5')

FileLink(r'finmodel.h5')

converter = tf.lite.TFLiteConverter.from_keras_model_file('finmodel.h5')

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

tflite_quant_model = converter.convert()

open ("modelquant.tflite" , "wb").write(tflite_quant_model)



FileLink(r'modelquant.tflite')