# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install git+https://github.com/aleju/imgaug.git
!git clone https://github.com/ahmadelsallab/MultiCheXNet.git
##!rm -r /kaggle/working/MultiCheXNet/
# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
def get_iou_vector(A, B):
    # Numpy version
    B = K.cast(B, 'float32')
    batch_size = A.shape[0]
    if batch_size is None:
      batch_size = 0
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            pred_batch_size = pred / ( p.shape[0] * p.shape[1] )
            if pred_batch_size > 0.03:
               pred_batch_size = 1 
            metric +=  1 - pred_batch_size
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels

        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_function(get_iou_vector, [label, pred > 0.5], tf.float64)

import tensorflow.keras.backend as K
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f
    score = (200. * K.sum(intersection) + smooth) / (100. * K.sum(y_true_f) + 100.* K.sum(y_pred_f) + smooth)
    return  (1. - score)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))



def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

from MultiCheXNet.data_loader.SIIM_ACR_dataloader import get_train_validation_generator
from MultiCheXNet.MTL_model import MTL_model
from tensorflow.keras.optimizers import Adam


seg_csv_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/train-rle.csv"
seg_images_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/dicom-images-train/"
train_gen,val_gen = get_train_validation_generator(seg_csv_path , seg_images_path, only_positive=False, augmentation=True,hist_eq=True,batch_positive_portion=0.5 )
X,Y = next(enumerate(train_gen))[1]
import matplotlib.pyplot as plt
import numpy as np
for yy in Y:
    plt.imshow(np.squeeze(yy))
    plt.show()

MTL_model_clss = MTL_model(add_class_head=False,add_detector_head=False,add_segmenter_head=True)
model = MTL_model_clss.MTL_model

lr=1e-4
epochs=10
model.compile(loss= binary_focal_loss(), optimizer=Adam(lr) , metrics=[ dice_coef , my_iou_metric])
hist= model.fit_generator(train_gen, validation_data=val_gen, epochs=epochs,class_weight={0:1,1:100} )
#0.1
#0.978
#0.77
#0.04
#0.93
#0.01













from tensorflow.keras.layers import *
def dense_block(x, blocks, name):
    #REF: keras-team
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x
def conv_block(x, growth_rate, name):
    #REF: keras-team
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def transition_up(x, encoder, skip_connection, out_channels, kernel_size=(3,3), stride=(2,2)):
    tu = Conv2DTranspose(out_channels, kernel_size, strides = stride, padding = 'same')(x)
    skip = encoder.layers[skip_connection].output
    c = concatenate([skip,tu], axis=3)
    return c
from MultiCheXNet.utils.Encoder import Encoder
from MultiCheXNet.utils.ModelBlock import ModelBlock
from MultiCheXNet.utils.Segmenter import Segmenter 
encoder_class = Encoder(weights=None)
seg_head= Segmenter(encoder_class)
model = ModelBlock.add_heads(encoder_class ,[seg_head] )
# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
from keras import backend as K
def get_iou_vector(A, B):
    # Numpy version
    B = K.cast(B, 'float32')
    batch_size = A.shape[0]
    if batch_size is None:
      batch_size = 0
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            pred_batch_size = pred / ( p.shape[0] * p.shape[1] )
            if pred_batch_size > 0.03:
               pred_batch_size = 1 
            metric +=  1 - pred_batch_size
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels

        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_function(get_iou_vector, [label, pred > 0.5], tf.float64)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return 100 * score
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


from MultiCheXNet.data_loader.SIIM_ACR_dataloader import get_train_validation_generator
seg_csv_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/train-rle.csv"
seg_images_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/dicom-images-train/"

seg_train_gen , seg_val_gen = get_train_validation_generator(seg_csv_path,seg_images_path ,augmentation=True,hist_eq=True,normalize=True )
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
model.compile(loss= "binary_crossentropy", optimizer=Adam(1e-4), metrics=[dice_coef,my_iou_metric])
model.fit_generator(seg_train_gen,validation_data=seg_val_gen , epochs=20)


skip_layers = [308,136,48]
blocks = [3, 3, 3, 3]

db5 = encoder.output #(8,8,1024)
tu5 = transition_up(db5, encoder, skip_layers[0], 3)

db6 = dense_block(tu5, blocks[-1], name='conv6')
tu6 = transition_up(db6, encoder, skip_layers[1], 3)

db7 = dense_block(tu6, blocks[-2], name='conv7')
tu7 = transition_up(db7, encoder, skip_layers[2], 3)

db8 = dense_block(tu7, blocks[-3], name='conv8')
tu8 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(db8)#(128,128,)

uconv9 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same')(tu8)
tu9 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same')(uconv9)#(256,256,)
outputs = Conv2D(1, (1, 1), activation = 'sigmoid')(tu9)




def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return 100 * score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f
    score = (200. * K.sum(intersection) + smooth) / (100. * K.sum(y_true_f) + 100.* K.sum(y_pred_f) + smooth)
    return  (1. - score)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))
from MultiCheXNet.data_loader.SIIM_ACR_dataloader import get_train_validation_generator

seg_csv_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/train-rle.csv"
seg_images_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/dicom-images-train/"

seg_train_gen , seg_val_gen = get_train_validation_generator(seg_csv_path,seg_images_path)
X,y = next(enumerate(seg_val_gen))[1]
X.shape,y.shape

from tensorflow.keras.optimizers import Adam
model.compile(loss= dice_loss, optimizer=Adam(1e-4), metrics=[dice_coef,my_iou_metric ])

import tensorflow as tf
import numpy as np

model.fit_generator(seg_train_gen , validation_data=seg_val_gen , epochs=10)
seg_csv_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/train-rle.csv"
seg_images_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/dicom-images-train/"

seg_train_gen , seg_val_gen = get_train_validation_generator(seg_csv_path,seg_images_path ,augmentation=True,hist_eq=True,normalize=True )


from MultiCheXNet.utils.Encoder import Encoder
from MultiCheXNet.utils.ModelBlock import ModelBlock
encoder_class = Encoder(weights=None)
encoder = encoder_class.model 
from tensorflow.keras.layers import Conv2DTranspose, Conv2D
encoder.summary()
X = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same',activation='relu')(encoder.output)#(16,16,)
X = Conv2D(512 , (1,1), padding='same',activation='relu')(X)
X = Conv2D(512 , (1,1), padding='same',activation='relu')(X)


X = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same',activation='relu')(X)#(32,32,)
X = Conv2D(256 , (1,1), padding='same',activation='relu')(X)
X = Conv2D(256 , (1,1), padding='same',activation='relu')(X)

X = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same',activation='relu')(X)#(64,64,)
X = Conv2D(128 , (1,1), padding='same',activation='relu')(X)
X = Conv2D(128 , (1,1), padding='same',activation='relu')(X)

X = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same',activation='relu')(X)#(128,128,)
X = Conv2D(128 , (1,1), padding='same',activation='relu')(X)
X = Conv2D(128 , (1,1), padding='same',activation='relu')(X)

X = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same',activation='relu')(X)#(256,256,)
X = Conv2D(32 , (1,1), padding='same',activation='relu')(X)
X = Conv2D(32 , (1,1), padding='same',activation='relu')(X)
X = Conv2D(1 , (1,1), padding='same',activation='sigmoid')(X)


model= ModelBlock.add_heads(encoder_class, [X] ,is_classes=False )
model.summary()
from tensorflow.keras.applications.densenet import preprocess_input
from MultiCheXNet.data_loader.SIIM_ACR_dataloader import get_train_validation_generator
seg_csv_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/train-rle.csv"
seg_images_path = "/kaggle/input/siim-acr-pneumothorax-segmentation-data/dicom-images-train/"

seg_train_gen , seg_val_gen = get_train_validation_generator(seg_csv_path,seg_images_path ,augmentation=True,hist_eq=True,normalize=True  )
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


model.compile(loss= "binary_crossentropy", optimizer=Adam(1e-5), metrics=[dice_coef,my_iou_metric ])

model.fit_generator(seg_train_gen , validation_data=seg_val_gen , epochs=10)
