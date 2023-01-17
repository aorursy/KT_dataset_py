# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.layers.core import Lambda, Flatten, Dense

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.models import Model

from keras import backend as K

import pandas as pd

import numpy as np

import keras

from keras.models import *

from keras.layers import *

from keras.preprocessing import *

from keras.callbacks import *

from keras.optimizers import *

from tqdm import tqdm

import glob

from keras.applications.resnet50 import preprocess_input, decode_predictions



# from keras.applications.nasnet import preprocess_input, decode_predictions

# from keras.applications.densenet import preprocess_input, decode_predictions

from keras.preprocessing import image

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import sys



import cv2

from random import choice, sample

import datetime

from sklearn.metrics import roc_auc_score

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import threading



from keras.initializers import glorot_normal



import tensorflow as tf

import numpy as np

import os



from numpy import genfromtxt

from keras.layers import Conv2D, ZeroPadding2D, Activation

from keras.layers.normalization import BatchNormalization


def outer_product(x):

    """

    calculate outer-products of 2 tensors



        args 

            x

                list of 2 tensors

                , assuming each of which has shape = (size_minibatch, total_pixels, size_filter)

    """

    return keras.backend.batch_dot(

                x[0]

                , x[1]

                , axes=[1,1]

            ) / x[0].get_shape().as_list()[1] 



def signed_sqrt(x):

    """

    calculate element-wise signed square root



        args

            x

                a tensor

    """

    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)



def L2_norm(x, axis=-1):

    """

    calculate L2-norm



        args 

            x

                a tensor

    """

    return keras.backend.l2_normalize(x, axis=axis)



def auc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)





        

def euclidean_distance(vects):

    x, y = vects

    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)

    return K.sqrt(K.maximum(sum_square, K.epsilon()))





def eucl_dist_output_shape(shapes):

    shape1, shape2 = shapes

    return (shape1[0], 1)





def contrastive_loss(y_true, y_pred):

    '''Contrastive loss from Hadsell-et-al.'06

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    '''

    margin = 1

    sqaure_pred = K.square(y_pred)

    margin_square = K.square(K.maximum(margin - y_pred, 0))

    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)



def create_base_network(input_shape):

    new_mod = Model(mod.input,mod.layers[-5].output)

    return new_mod



def compute_accuracy(y_true, y_pred):

    '''Compute classification accuracy with a fixed threshold on distances.

    '''

    pred = y_pred.ravel() < 0.5

    return np.mean(pred == y_true)





def accuracy(y_true, y_pred):

    '''Compute classification accuracy with a fixed threshold on distances.

    '''

    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))









def LRN2D(x):

    return tf.nn.lrn(x, alpha=1e-4, beta=0.75)



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


ROOT = "../input/recognizing-faces-in-the-wild/"





def read_img(im_path):

    im1 = image.load_img(im_path, target_size=(96, 96))

    im1 = image.img_to_array(im1)

    im1 = np.expand_dims(im1, axis=0)

    im1 = preprocess_input(im1,mode='tf')

    return im1[0]



from collections import defaultdict

#keeps all photos path in a dictionary

allPhotos = defaultdict(list)

for family in glob.glob(ROOT+"train/*"):

    for mem in glob.glob(family+'/*'):

        for photo in glob.glob(mem+'/*'):

            allPhotos[mem].append(photo)



#list of all members with valid photo

ppl = list(allPhotos.keys())

len(ppl)



data = pd.read_csv(ROOT+'train_relationships.csv')

data.p1 = data.p1.apply( lambda x: ROOT+'train/'+x )

data.p2 = data.p2.apply( lambda x: ROOT+'train/'+x )

print(data.shape)

data.head()



data = data[ ( (data.p1.isin(ppl)) & (data.p2.isin(ppl)) ) ]

data = [ ( x[0], x[1]  ) for x in data.values ]

len(data)



train = [ x for x in data if 'F09' not in x[0]  ]

val = [ x for x in data if 'F09' in x[0]  ]

len(train), len(val)



def getImages(p1,p2):

    p1 = read_img(choice(allPhotos[p1]))

    p2 = read_img(choice(allPhotos[p2]))

    return p1,p2



def getMiniBatch(batch_size=16, data=train):

    p1 = []; p2 = []; Y = []

    batch = sample(data, batch_size//2)

    for x in batch:

        _p1, _p2 = getImages(*x)

        p1.append(_p1);p2.append(_p2);Y.append(1)

    while len(Y) < batch_size:

        _p1,_p2 = tuple(np.random.choice(ppl,size=2, replace=False))

        if (_p1,_p2) not in train+val and (_p2,_p1) not in train+val:

            _p1,_p2 = getImages(_p1,_p2)

            p1.append(_p1);p2.append(_p2);Y.append(0) 

    return [np.array(p1),np.array(p2)], np.array(Y)



def create_model():

    myInput = Input(shape=(96, 96, 3))



    x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)

    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)

    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)

    x = MaxPooling2D(pool_size=3, strides=2)(x)

    x = Lambda(LRN2D, name='lrn_1')(x)

    x = Conv2D(64, (1, 1), name='conv2')(x)

    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)

    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1))(x)

    x = Conv2D(192, (3, 3), name='conv3')(x)

    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)

    x = Activation('relu')(x)

    x = Lambda(LRN2D, name='lrn_2')(x)

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



    inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3a)

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



    inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3c)

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



    inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4e)

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



    return Model(inputs=[myInput], outputs=norm_layer)







mod = create_model()

mod.load_weights('../input/kin-detection/nn4.small2.v1.h5')

ModelCheckpoint('model_best_checkpoint.h5', save_best_only=True,

                                    save_weights_only=True, monitor='val_auc', mode='max', verbose=1)
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',

                           patience=8,

                           verbose=1,

                           min_delta=1e-4),

             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',

                               factor=0.1,

                               patience=4,

                               verbose=1,

                               epsilon=1e-4),

            keras.callbacks.ModelCheckpoint('model_best_checkpoint.h5',save_best_only=True,

                                            verbose=0, save_weights_only=True,monitor='val_auc',mode='max'),

        ]


input_shape = (96,96,3)



input_a = Input(shape=input_shape)

input_b = Input(shape=input_shape)



base_network = create_base_network(input_shape)



# because we re-use the same instance `base_network`,

# the weights of the network

# will be shared across the two branches

processed_a = base_network(input_a)

processed_b = base_network(input_b)





# extract features from detector

x_detector = processed_a

shape_detector = processed_a.shape



# extract features from extractor , same with detector for symmetry DxD model

shape_extractor = processed_b.shape

x_extractor = processed_b



# rehape to (minibatch_size, total_pixels, filter_size)

x_detector = keras.layers.Reshape([shape_detector[1] * shape_detector[2] , shape_detector[-1]])(x_detector)

x_extractor = keras.layers.Reshape([shape_extractor[1] * shape_extractor[2] , shape_extractor[-1]])(x_extractor)



# outer products of features, output shape=(minibatch_size, filter_size_detector*filter_size_extractor)

x = keras.layers.Lambda(outer_product)([x_detector, x_extractor])



# rehape to (minibatch_size, filter_size_detector*filter_size_extractor)

x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)

# signed square-root 



x = keras.layers.Lambda(signed_sqrt)(x)

# L2 normalization

x = keras.layers.Lambda(L2_norm)(x)



### 

### attach FC-Layer

###



x = Dense(100, activation="relu")(x)

x = Dropout(0.01)(x)



out = Dense(units=1,kernel_regularizer=keras.regularizers.l2(1e-8),kernel_initializer='glorot_normal',activation="sigmoid")(x)







model = Model([input_a, input_b], out)



model.compile(loss="binary_crossentropy", metrics=['acc',auc], optimizer=Adam(0.001))





# model.load_weights("/home/datumx/data_science_experiments/detect_kinship/logs/kin_relation_2019_05_28_21_53_51966472/siamese_kins_detection_13-val_loss_0.4720-val_acc_0.7680.h5")

# train





print(model.summary())

# train_batches =batch_generator(data_train,batch_size=8)

# valid_batches =batch_generator(data_valid,batch_size=8)



def Generator(batch_size, data ):

    while True:

        yield getMiniBatch(batch_size=batch_size, data=data)



train_gen = Generator(batch_size=16,data=train)

val_gen = Generator(batch_size=16,data=val)



model.fit_generator(train_gen,steps_per_epoch=200,use_multiprocessing=True,

          epochs=100,validation_data=val_gen,validation_steps=100,callbacks=callbacks)

test_path = "../input/recognizing-faces-in-the-wild/test/"
model.load_weights('model_best_checkpoint.h5')

def chunker(seq, size=32):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



from tqdm import tqdm



submission = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')



predictions = []
predictions = []

for batch in tqdm(chunker(submission.img_pair.values)):

    X1 = [x.split("-")[0] for x in batch]

    X1 = np.array([read_img(test_path + x) for x in X1])



    X2 = [x.split("-")[1] for x in batch]

    X2 = np.array([read_img(test_path + x) for x in X2])



    pred = model.predict([X1, X2]).ravel().tolist()

    predictions += pred
submission['is_related'] = predictions



submission.to_csv("sub_detect_kinship.csv", index=False)