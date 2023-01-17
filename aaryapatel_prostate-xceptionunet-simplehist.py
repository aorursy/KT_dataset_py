!pip install SimpleITK



import numpy as np

import os

import time

import pandas as pd

import fnmatch

import cv2

import matplotlib.pyplot as plt

import SimpleITK as sitk

from skimage.exposure import equalize_adapthist, equalize_hist



!pip install albumentations > /dev/null

!pip install -U segmentation-models

#!pip install -U efficientnet==0.0.4

import numpy as np

import pandas as pd

import gc

import keras



import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

import seaborn as sns

sns.set_style("white")



from sklearn.model_selection import train_test_split,StratifiedKFold



from skimage.transform import resize

import tensorflow as tf

import keras.backend as K

from keras.losses import binary_crossentropy



from keras.preprocessing.image import load_img

from keras import Model

from keras.callbacks import  ModelCheckpoint

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, BatchNormalization

from keras.layers import Conv2D, Concatenate, MaxPooling2D

from keras.layers import UpSampling2D, Dropout, BatchNormalization

from tqdm import tqdm_notebook

from keras import initializers

from keras import regularizers

from keras import constraints

from keras.utils import conv_utils

from keras.utils.data_utils import get_file

from keras.engine.topology import get_source_inputs

from keras.engine import InputSpec

from keras import backend as K

from keras.layers import LeakyReLU

from keras.layers import ZeroPadding2D

from keras.losses import binary_crossentropy

import keras.callbacks as callbacks

from keras.callbacks import Callback

from keras.applications.xception import Xception

from keras.layers import multiply





from keras import optimizers

from keras.legacy import interfaces

from keras.utils.generic_utils import get_custom_objects



from keras.engine.topology import Input

from keras.engine.training import Model

from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose

from keras.layers.core import Activation, SpatialDropout2D

from keras.layers.merge import concatenate

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D

from keras.layers import Input,Dropout,BatchNormalization,Activation,Add

from keras.regularizers import l2

from keras.layers.core import Dense, Lambda

from keras.layers.merge import concatenate, add

from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator



import glob

import shutil

import os

import random

from PIL import Image

import cv2

seed = 10

np.random.seed(seed)

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

#tf.set_random_seed(seed)

    

%matplotlib inline

from keras.models import Model

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose

from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D, BatchNormalization

from keras import backend as K

from keras.optimizers import *

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator



from functools import partial

from keras.initializers import RandomNormal, VarianceScaling

import numpy as np



print(tf.__version__)
def load_data():

  return np.load('../input/clahe-001/X_train(SimpleHist).npy'), np.load('../input/clahe-001/y_train(SimpleHist).npy')
X_data, y_data = load_data()
from sklearn.model_selection import train_test_split

#from wandb import magic
# Evalaution Metrics

def dice_coef(y_true, y_pred, smooth=1.0):



    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (

        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)





def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score
def elastic_transform(image, x=None, y=None, alpha=256*3, sigma=256*0.07):

    """Elastic deformation of images as described in [Simard2003]_.

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for

       Convolutional Neural Networks applied to Visual Document Analysis", in

       Proc. of the International Conference on Document Analysis and

       Recognition, 2003.

    """



    shape = image.shape

    blur_size = int(4*sigma) | 1

    dx = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha

    dy = cv2.GaussianBlur((np.random.rand(shape[0],shape[1]) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma)* alpha



    if (x is None) or (y is None):

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')



    map_x =  (x+dx).astype('float32')

    map_y =  (y+dy).astype('float32')



    return cv2.remap(image.astype('float32'), map_y,  map_x, interpolation=cv2.INTER_NEAREST).reshape(shape)
class SnapshotCallbackBuilder:

    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.001):

        self.T = nb_epochs

        self.M = nb_snapshots

        self.alpha_zero = init_lr



    def get_callbacks(self, model_prefix='Model'):



        callback_list = [

            callbacks.ModelCheckpoint("best_xception_model_simplehist.h5",monitor='val_dice_coef', 

                                   mode = 'max', save_best_only=True, verbose=1),

            swa,

            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)

        ]



        return callback_list



    def _cosine_anneal_schedule(self, t):

        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.

        cos_inner /= self.T // self.M

        cos_out = np.cos(cos_inner) + 1

        return float(self.alpha_zero / 2 * cos_out)
class SWA(keras.callbacks.Callback):

    

    def __init__(self, filepath, swa_epoch):

        super(SWA, self).__init__()

        self.filepath = filepath

        self.swa_epoch = swa_epoch 

    

    def on_train_begin(self, logs=None):

        self.nb_epoch = self.params['epochs']

        print('Stochastic weight averaging selected for last {} epochs.'

              .format(self.nb_epoch - self.swa_epoch))

        

    def on_epoch_end(self, epoch, logs=None):

        

        if epoch == self.swa_epoch:

            self.swa_weights = self.model.get_weights()

            

        elif epoch > self.swa_epoch:    

            for i in range(len(self.swa_weights)):

                self.swa_weights[i] = (self.swa_weights[i] * 

                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  



        else:

            pass

        

    def on_train_end(self, logs=None):

        self.model.set_weights(self.swa_weights)

        print('Final model parameters set to stochastic weight average.')

        self.model.save_weights(self.filepath)

        print('Final stochastic averaged weights saved to file.')
def keras_fit_generator(img_rows=256, img_cols=256, n_imgs=15 * 10 ** 4, batch_size=32, epochs = 50, regenerate=True):



    # Data-split

    X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.1)



    # img_rows = X_train.shape[1]

    # img_cols =  X_train.shape[2]



    # Provide the same seed and keyword arguments to the fit and flow methods



    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')

    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows*1.5, sigma=img_rows*0.07 )

    # we create two instances with the same arguments

    data_gen_args = dict(

        featurewise_center=False,

        featurewise_std_normalization=False,

        rotation_range=10.,

        width_shift_range=0.1,

        height_shift_range=0.1,

        horizontal_flip=True,

        vertical_flip=True,

        zoom_range=[1, 1.2],

        fill_mode='constant',

        preprocessing_function=elastic)



    image_datagen = ImageDataGenerator(**data_gen_args)

    mask_datagen = ImageDataGenerator(**data_gen_args)



    seed = 2

    image_datagen.fit(X_train, seed=seed)

    mask_datagen.fit(y_train, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)



    model = UXception(input_shape=(img_rows, img_cols, 1))

    model.load_weights('../input/xception-simplehist/best_xception_model_weights_simplehist.h5')



   # model.summary()

    

    # model_checkpoint = ModelCheckpoint(

    #     'model_weights_5.h5', monitor='val_loss', save_best_only=True)



    # c_backs = [model_checkpoint,swa]

    # c_backs.append( EarlyStopping(monitor='loss', min_delta=0.001, patience=5) )



    model.compile(  optimizer=Adam(lr=1e-3), loss=dice_loss, metrics=[dice_coef])



    history = model.fit_generator(

                        train_generator,

                        steps_per_epoch=n_imgs//batch_size,

                        epochs=epochs,

                        shuffle=True,

                        validation_data=(X_test, y_test),

                        callbacks=snapshot.get_callbacks())

    

    plt.figure(figsize=(16,4))

    plt.subplot(1,2,1)

    plt.plot(history.history['dice_coef'][1:])

    plt.plot(history.history['val_dice_coef'][1:])

    plt.ylabel('dice coefficient')

    plt.xlabel('epoch')

    plt.legend(['train','Validation'], loc='upper left')



    plt.title('model Dice Coefficient')

    plt.savefig('xception_dice.png')



    plt.subplot(1,2,2)

    plt.plot(history.history['loss'][1:])

    plt.plot(history.history['val_loss'][1:])

    plt.ylabel('loss value')

    plt.xlabel('number of epochs')

    plt.legend(['train','Validation'], loc='upper left')

    plt.title('model loss')

    plt.savefig('xception_loss.png')

    

    pd.DataFrame(history.history).to_hdf("xception_hist.h5",key="history")
def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):

    x = Conv2D(filters, size, strides=strides, padding=padding)(x)

    x = BatchNormalization()(x)

    if activation == True:

        x = LeakyReLU(alpha=0.1)(x)

    return x



def residual_block(blockInput, num_filters=16):

    x = LeakyReLU(alpha=0.1)(blockInput)

    x = BatchNormalization()(x)

    blockInput = BatchNormalization()(blockInput)

    x = convolution_block(x, num_filters, (3,3))

    x = convolution_block(x, num_filters, (3,3), activation=False)

    x = Add()([x, blockInput])

    return x
def UXception(input_shape=(None, None, 3),dropout_rate=0.5):



    backbone = Xception(input_shape=input_shape,weights=None,include_top=False)

    input = backbone.input

    start_neurons = 16



    conv4 = backbone.layers[121].output

    conv4 = LeakyReLU(alpha=0.1)(conv4)

    pool4 = MaxPooling2D((2, 2))(conv4)

    pool4 = Dropout(dropout_rate)(pool4)

    

     # Middle

    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)

    convm = residual_block(convm,start_neurons * 32)

    convm = residual_block(convm,start_neurons * 32)

    convm = LeakyReLU(alpha=0.1)(convm)

    

    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)

    uconv4 = concatenate([deconv4, conv4])

    uconv4 = Dropout(dropout_rate)(uconv4)

    

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)

    uconv4 = residual_block(uconv4,start_neurons * 16)

    uconv4 = residual_block(uconv4,start_neurons * 16)

    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)

    conv3 = backbone.layers[31].output

    uconv3 = concatenate([deconv3, conv3])    

    uconv3 = Dropout(dropout_rate)(uconv3)

    

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)

    uconv3 = residual_block(uconv3,start_neurons * 8)

    uconv3 = residual_block(uconv3,start_neurons * 8)

    uconv3 = LeakyReLU(alpha=0.1)(uconv3)



    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)

    conv2 = backbone.layers[21].output

    conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)

    uconv2 = concatenate([deconv2, conv2])

        

    uconv2 = Dropout(0.1)(uconv2)

    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)

    uconv2 = residual_block(uconv2,start_neurons * 4)

    uconv2 = residual_block(uconv2,start_neurons * 4)

    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    

    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)

    conv1 = backbone.layers[11].output

    conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1)

    uconv1 = concatenate([deconv1, conv1])

    

    uconv1 = Dropout(0.1)(uconv1)

    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)

    uconv1 = residual_block(uconv1,start_neurons * 2)

    uconv1 = residual_block(uconv1,start_neurons * 2)

    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   

    uconv0 = Dropout(dropout_rate)(uconv0)

    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)

    uconv0 = residual_block(uconv0,start_neurons * 1)

    uconv0 = residual_block(uconv0,start_neurons * 1)

    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    

    uconv0 = Dropout(dropout_rate/2)(uconv0)

    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv0)    

    

    model = Model(input, output_layer)

    model.name = 'u-xception'



    return model
model = UXception((256,256,1))

model.summary()
import time

epochs = 4

swa = SWA('best_xception_model_weights_simplehist.h5',epochs - 1)

snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-4)

start = time.time()

keras_fit_generator(img_rows=256, img_cols=256, regenerate=True,

                     batch_size=16,epochs = epochs)



end = time.time()