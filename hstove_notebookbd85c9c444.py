# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# print(check_output(["ls", "../input/train"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import shutil

import random



new_dir = "~/hank/"



try:

    shutil.rmtree(new_dir)

except FileNotFoundError:

    None



train_files = os.listdir("../input/train")

cat_files = [file for file in train_files if "cat" in file]

dog_files = [file for file in train_files if "dog" in file]



random.shuffle(dog_files)

random.shuffle(cat_files)



train_split = 0.8

train_idx = round(len(dog_files) * train_split)



dog_train = dog_files[:train_idx]

dog_valid = dog_files[train_idx:]



cat_train = cat_files[:train_idx]

cat_valid = cat_files[train_idx:]



sample_size = 10

dog_sample = dog_files[:sample_size]

cat_sample = cat_files[:sample_size]
total, used, free = shutil.disk_usage("/")

print ("{} free mb".format(free / 1000000))
import warnings

import numpy as np



from keras.models import Model

from keras.layers import Dense, Input, BatchNormalization, Activation, merge

from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.preprocessing import image

from keras.utils.data_utils import get_file

from keras import backend as K
# dog_valid_imgs = [image.load_img("../input/train/"+file, target_size=(299, 299)) for file in dog_valid]

# cat_valid_imgs = [image.load_img("../input/train/"+file, target_size=(299, 299)) for file in cat_valid]

# dog_train_imgs = [image.load_img("../input/train/"+file, target_size=(299, 299)) for file in dog_train]

# cat_train_imgs = [image.load_img("../input/train/"+file, target_size=(299, 299)) for file in cat_train]

# dog_sample_imgs = [image.load_img("../input/train/"+file, target_size=(299, 299)) for file in dog_sample]

# cat_sample_imgs = [image.load_img("../input/train/"+file, target_size=(299, 299)) for file in cat_sample]
# TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'

# TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'

TF_WEIGHTS_PATH = "https://s3.amazonaws.com/deep-learning-weights/xception_weights_tf_dim_ordering_tf_kernels.h5"

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
def Xception(include_top=True, weights='imagenet',

             input_tensor=None):

    '''Instantiate the Xception architecture,

    optionally loading weights pre-trained

    on ImageNet. This model is available for TensorFlow only,

    and can only be used with inputs following the TensorFlow

    dimension ordering `(width, height, channels)`.

    You should set `image_dim_ordering="tf"` in your Keras config

    located at ~/.keras/keras.json.

    Note that the default input image size for this model is 299x299.

    # Arguments

        include_top: whether to include the fully-connected

            layer at the top of the network.

        weights: one of `None` (random initialization)

            or "imagenet" (pre-training on ImageNet).

        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)

            to use as image input for the model.

    # Returns

        A Keras model instance.

    '''

    if weights not in {'imagenet', None}:

        raise ValueError('The `weights` argument should be either '

                         '`None` (random initialization) or `imagenet` '

                         '(pre-training on ImageNet).')

    if K.backend() != 'tensorflow':

        raise Exception('The Xception model is only available with '

                        'the TensorFlow backend.')

    if K.image_dim_ordering() != 'tf':

        warnings.warn('The Xception model is only available for the '

                      'input dimension ordering "tf" '

                      '(width, height, channels). '

                      'However your settings specify the default '

                      'dimension ordering "th" (channels, width, height). '

                      'You should set `image_dim_ordering="tf"` in your Keras '

                      'config located at ~/.keras/keras.json. '

                      'The model being returned right now will expect inputs '

                      'to follow the "tf" dimension ordering.')

        K.set_image_dim_ordering('tf')

        old_dim_ordering = 'th'

    else:

        old_dim_ordering = None



    # Determine proper input shape

    if include_top:

        input_shape = (299, 299, 3)

    else:

        input_shape = (None, None, 3)



    if input_tensor is None:

        img_input = Input(shape=input_shape)

    else:

        if not K.is_keras_tensor(input_tensor):

            img_input = Input(tensor=input_tensor, shape=input_shape)

        else:

            img_input = input_tensor



    x = Conv2D(32, 3, 3, subsample=(2, 2), bias=False, name='block1_conv1')(img_input)

    x = BatchNormalization(name='block1_conv1_bn')(x)

    x = Activation('relu', name='block1_conv1_act')(x)

    x = Conv2D(64, 3, 3, bias=False, name='block1_conv2')(x)

    x = BatchNormalization(name='block1_conv2_bn')(x)

    x = Activation('relu', name='block1_conv2_act')(x)



    residual = Conv2D(128, 1, 1, subsample=(2, 2),

                      border_mode='same', bias=False)(x)

    residual = BatchNormalization()(residual)



    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv1')(x)

    x = BatchNormalization(name='block2_sepconv1_bn')(x)

    x = Activation('relu', name='block2_sepconv2_act')(x)

    x = SeparableConv2D(128, 3, 3, border_mode='same', bias=False, name='block2_sepconv2')(x)

    x = BatchNormalization(name='block2_sepconv2_bn')(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)

    x = merge([x, residual], mode='sum')



    residual = Conv2D(256, 1, 1, subsample=(2, 2),

                      border_mode='same', bias=False)(x)

    residual = BatchNormalization()(residual)



    x = Activation('relu', name='block3_sepconv1_act')(x)

    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv1')(x)

    x = BatchNormalization(name='block3_sepconv1_bn')(x)

    x = Activation('relu', name='block3_sepconv2_act')(x)

    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv2')(x)

    x = BatchNormalization(name='block3_sepconv2_bn')(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)

    x = merge([x, residual], mode='sum')



    residual = Conv2D(728, 1, 1, subsample=(2, 2),

                      border_mode='same', bias=False)(x)

    residual = BatchNormalization()(residual)



    x = Activation('relu', name='block4_sepconv1_act')(x)

    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv1')(x)

    x = BatchNormalization(name='block4_sepconv1_bn')(x)

    x = Activation('relu', name='block4_sepconv2_act')(x)

    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv2')(x)

    x = BatchNormalization(name='block4_sepconv2_bn')(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)

    x = merge([x, residual], mode='sum')



    for i in range(8):

        residual = x

        prefix = 'block' + str(i + 5)



        x = Activation('relu', name=prefix + '_sepconv1_act')(x)

        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1')(x)

        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)

        x = Activation('relu', name=prefix + '_sepconv2_act')(x)

        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2')(x)

        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)

        x = Activation('relu', name=prefix + '_sepconv3_act')(x)

        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3')(x)

        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)



        x = merge([x, residual], mode='sum')



    residual = Conv2D(1024, 1, 1, subsample=(2, 2),

                      border_mode='same', bias=False)(x)

    residual = BatchNormalization()(residual)



    x = Activation('relu', name='block13_sepconv1_act')(x)

    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block13_sepconv1')(x)

    x = BatchNormalization(name='block13_sepconv1_bn')(x)

    x = Activation('relu', name='block13_sepconv2_act')(x)

    x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block13_sepconv2')(x)

    x = BatchNormalization(name='block13_sepconv2_bn')(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block13_pool')(x)

    x = merge([x, residual], mode='sum')



    x = SeparableConv2D(1536, 3, 3, border_mode='same', bias=False, name='block14_sepconv1')(x)

    x = BatchNormalization(name='block14_sepconv1_bn')(x)

    x = Activation('relu', name='block14_sepconv1_act')(x)



    x = SeparableConv2D(2048, 3, 3, border_mode='same', bias=False, name='block14_sepconv2')(x)

    x = BatchNormalization(name='block14_sepconv2_bn')(x)

    x = Activation('relu', name='block14_sepconv2_act')(x)



    if include_top:

        x = GlobalAveragePooling2D(name='avg_pool')(x)

        x = Dense(1000, activation='softmax', name='predictions')(x)



    # Create model

    model = Model(img_input, x)



    # load weights

    if weights == 'imagenet':

        if include_top:

            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',

                                    TF_WEIGHTS_PATH,

                                    cache_subdir='models')

        else:

            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',

                                    TF_WEIGHTS_PATH_NO_TOP,

                                    cache_subdir='models')

        model.load_weights(weights_path)



    if old_dim_ordering:

        K.set_image_dim_ordering(old_dim_ordering)

    return model
def preprocess_input(x):

    x /= 255.

    x -= 0.5

    x *= 2.

    return x
generator = image.ImageDataGenerator()
def get_imgs(arr):

    imgs = [image.load_img("../input/train/"+file, target_size=(299, 299)) for file in arr]

    imgs = [preprocess_input(np.expand_dims(image.img_to_array(img), axis=0)) for img in imgs]

    return imgs
dog_sample_imgs = get_imgs(dog_sample)

cat_sample_imgs = get_imgs(cat_sample)
X_sample = dog_sample_imgs + cat_sample_imgs
Y_sample = [0] * len(dog_sample_imgs)

Y_sample += [1] * len(cat_sample_imgs)
model = Xception()