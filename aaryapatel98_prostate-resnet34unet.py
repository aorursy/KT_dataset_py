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

  return np.load('../input/prostate/X_train(Hist).npy'), np.load('../input/prostate/y_train(Hist).npy')
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

            callbacks.ModelCheckpoint("best_resnet34_model.h5",monitor='val_dice_coef', 

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
import keras

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



    model = sm.Unet('resnet34', input_shape=(img_rows, img_cols, 1), encoder_weights=None)

    model.load_weights('../input/resnet34prostate/best_resnet34_model.h5')



   # model.summary()

    

    # model_checkpoint = ModelCheckpoint(

    #     'model_weights_5.h5', monitor='val_loss', save_best_only=True)



    # c_backs = [model_checkpoint,swa]

    # c_backs.append( EarlyStopping(monitor='loss', min_delta=0.001, patience=5) )



    model.compile(optimizer=Adam(lr=1e-3), loss=dice_loss, metrics=[dice_coef])



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

    plt.xlabel('number of epochs')

    plt.legend(['train','Validation'], loc='upper left')



    plt.title('model Dice Coefficient')

    plt.savefig('resnet34_dice.png')



    plt.subplot(1,2,2)

    plt.plot(history.history['loss'][1:])

    plt.plot(history.history['val_loss'][1:])

    plt.ylabel('loss value')

    plt.xlabel('number of epochs')

    plt.legend(['train','Validation'], loc='upper left')

    plt.title('model loss')

    plt.savefig('resnet34_loss.png')



    pd.DataFrame(history.history).to_hdf("Resnet34_hist.h5",key="history")
import segmentation_models as sm

model = sm.Unet('resnet34', input_shape=(None, None, 1), encoder_weights=None)
import time

epochs = 6

swa = SWA('best_resnet34_model_weights.h5',epochs - 1)

snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)

start = time.time()

keras_fit_generator(img_rows=256, img_cols=256, regenerate=True,

                     batch_size=64,epochs = epochs)



end = time.time()