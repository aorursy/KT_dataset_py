# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io



import bson                       # this is installed with the pymongo package

from skimage.data import imread   # or, whatever image library you prefer

import multiprocessing as mp      # will come in handy due to the size of the data

import time

import datetime as dt



import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import concurrent.futures

from multiprocessing import cpu_count





from keras.optimizers import SGD

from keras import layers

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation

from keras.layers import GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import backend as K

from keras.utils.data_utils import get_file



from sklearn.metrics import log_loss

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
INPUT_PATH  = '../input/'

train_data_dir      = '../output/train'

validation_data_dir = '../output/validation'



WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'





img_width  = 128

img_height = 128

batch_size = 300
# the generators do the reading of input files

# If you want to use image augmentation use this configuration. For more choices and details

# consult the Keras manual: https://keras.io/preprocessing/image/



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)

print('Time right now: ', dt.datetime.now())

start = time.time()



# As we get familiar with this project, we forgo image augmentation, 

# we will later set shuffle of the samples to true as the optimizer of the fit is SGD



train_generator = test_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode="categorical",

    shuffle=False)

end = time.time()

print('Time right now: ', dt.datetime.now())

print('It took ', end-start, ' seconds to prepare train gen')

start = time.time()

validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode="categorical",

    shuffle=False)



end = time.time()



print('It took ', end-start, ' seconds to prepare valid gen')
img_rows = 128

img_cols = 128

num_classes = 5270

channel = 3

#from https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py

def conv2d_bn(x,

              filters,

              num_row,

              num_col,

              padding='same',

              strides=(1, 1),

              name=None):

    """Utility function to apply conv + BN.

    Arguments:

        x: input tensor.

        filters: filters in `Conv2D`.

        num_row: height of the convolution kernel.

        num_col: width of the convolution kernel.

        padding: padding mode in `Conv2D`.

        strides: strides in `Conv2D`.

        name: name of the ops; will become `name + '_conv'`

            for the convolution and `name + '_bn'` for the

            batch norm layer.

    Returns:

        Output tensor after applying `Conv2D` and `BatchNormalization`.

    """

    if name is not None:

        bn_name = name + '_bn'

        conv_name = name + '_conv'

    else:

        bn_name = None

        conv_name = None

    if K.image_data_format() == 'channels_first':

        bn_axis = 1

    else:

        bn_axis = 3

    x = Conv2D(

        filters, (num_row, num_col),

        strides=strides,

        padding=padding,

        use_bias=False,

        name=conv_name)(x)

    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    x = Activation('relu', name=name)(x)

    return x





def InceptionV3(include_top=True,

                weights='imagenet',

                input_tensor=None,

                input_shape=None,

                pooling=None,

                classes=1000):

    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained

    on ImageNet. Note that when using TensorFlow,

    for best performance you should set

    `image_data_format="channels_last"` in your Keras config

    at ~/.keras/keras.json.

    The model and the weights are compatible with both

    TensorFlow and Theano. The data format

    convention used by the model is the one

    specified in your Keras config file.

    Note that the default input image size for this model is 299x299.

    Arguments:

        include_top: whether to include the fully-connected

            layer at the top of the network.

        weights: one of `None` (random initialization)

            or "imagenet" (pre-training on ImageNet).

        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)

            to use as image input for the model.

        input_shape: optional shape tuple, only to be specified

            if `include_top` is False (otherwise the input shape

            has to be `(299, 299, 3)` (with `channels_last` data format)

            or `(3, 299, 299)` (with `channels_first` data format).

            It should have exactly 3 inputs channels,

            and width and height should be no smaller than 139.

            E.g. `(150, 150, 3)` would be one valid value.

        pooling: Optional pooling mode for feature extraction

            when `include_top` is `False`.

            - `None` means that the output of the model will be

                the 4D tensor output of the

                last convolutional layer.

            - `avg` means that global average pooling

                will be applied to the output of the

                last convolutional layer, and thus

                the output of the model will be a 2D tensor.

            - `max` means that global max pooling will

                be applied.

        classes: optional number of classes to classify images

            into, only to be specified if `include_top` is True, and

            if no `weights` argument is specified.

    Returns:

        A Keras model instance.

    Raises:

        ValueError: in case of invalid argument for `weights`,

            or invalid input shape.

    """

    if weights not in {'imagenet', None}:

        raise ValueError('The `weights` argument should be either '

                         '`None` (random initialization) or `imagenet` '

                         '(pre-training on ImageNet).')



    if weights == 'imagenet' and include_top and classes != 1000:

        raise ValueError('If using `weights` as imagenet with `include_top`'

                         ' as true, `classes` should be 1000')



    # Determine proper input shape

    #input_shape = _obtain_input_shape(

    #    input_shape,

    #    default_size=299,

    #    min_size=139,

    #    data_format=K.image_data_format(),

    #    include_top=include_top)



    if input_tensor is None:

        #img_input = Input(shape=input_shape)

        img_input = Input(shape=(img_rows, img_cols, channel))

    else:

        img_input = Input(tensor=input_tensor, shape=input_shape)



    if K.image_data_format() == 'channels_first':

        channel_axis = 1

        print('Theano')

    else:

        channel_axis = 3

        print('TensorFlow')



    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')

    x = conv2d_bn(x, 32, 3, 3, padding='valid')

    x = conv2d_bn(x, 64, 3, 3)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)



    x = conv2d_bn(x, 80, 1, 1, padding='valid')

    x = conv2d_bn(x, 192, 3, 3, padding='valid')

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)



    # mixed 0, 1, 2: 35 x 35 x 256

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed0')



    # mixed 1: 35 x 35 x 256

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed1')



    # mixed 2: 35 x 35 x 256

    branch1x1 = conv2d_bn(x, 64, 1, 1)



    branch5x5 = conv2d_bn(x, 48, 1, 1)

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)



    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch5x5, branch3x3dbl, branch_pool],

        axis=channel_axis,

        name='mixed2')



    # mixed 3: 17 x 17 x 768

    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')



    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch3x3dbl = conv2d_bn(

        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')



    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.concatenate(

        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')



    # mixed 4: 17 x 17 x 768

    branch1x1 = conv2d_bn(x, 192, 1, 1)



    branch7x7 = conv2d_bn(x, 128, 1, 1)

    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



    branch7x7dbl = conv2d_bn(x, 128, 1, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch7x7, branch7x7dbl, branch_pool],

        axis=channel_axis,

        name='mixed4')



    # mixed 5, 6: 17 x 17 x 768

    for i in range(2):

        branch1x1 = conv2d_bn(x, 192, 1, 1)



        branch7x7 = conv2d_bn(x, 160, 1, 1)

        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)

        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



        branch7x7dbl = conv2d_bn(x, 160, 1, 1)

        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)

        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)

        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)

        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



        branch_pool = AveragePooling2D(

            (3, 3), strides=(1, 1), padding='same')(x)

        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate(

            [branch1x1, branch7x7, branch7x7dbl, branch_pool],

            axis=channel_axis,

            name='mixed' + str(5 + i))



    # mixed 7: 17 x 17 x 768

    branch1x1 = conv2d_bn(x, 192, 1, 1)



    branch7x7 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)



    branch7x7dbl = conv2d_bn(x, 192, 1, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)



    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    x = layers.concatenate(

        [branch1x1, branch7x7, branch7x7dbl, branch_pool],

        axis=channel_axis,

        name='mixed7')



    # mixed 8: 8 x 8 x 1280

    branch3x3 = conv2d_bn(x, 192, 1, 1)

    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,

                          strides=(2, 2), padding='valid')



    branch7x7x3 = conv2d_bn(x, 192, 1, 1)

    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)

    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)

    branch7x7x3 = conv2d_bn(

        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')



    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = layers.concatenate(

        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')



    # mixed 9: 8 x 8 x 2048

    for i in range(2):

        branch1x1 = conv2d_bn(x, 320, 1, 1)



        branch3x3 = conv2d_bn(x, 384, 1, 1)

        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)

        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)

        branch3x3 = layers.concatenate(

            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))



        branch3x3dbl = conv2d_bn(x, 448, 1, 1)

        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)

        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)

        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)

        branch3x3dbl = layers.concatenate(

            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)



        branch_pool = AveragePooling2D(

            (3, 3), strides=(1, 1), padding='same')(x)

        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate(

            [branch1x1, branch3x3, branch3x3dbl, branch_pool],

            axis=channel_axis,

            name='mixed' + str(9 + i))

    if include_top:

        # Classification block

        x_fc = GlobalAveragePooling2D(name='avg_pool')(x)

        x_fc = Dense(classes, activation='softmax', name='predictions')(x_fc)

    else:

        if pooling == 'avg':

            x_fc = GlobalAveragePooling2D()(x_fc)

        elif pooling == 'max':

            x_fc = GlobalMaxPooling2D()(x_fc)



    # Ensure that the model takes into account

    # any potential predecessors of `input_tensor`.

    if input_tensor is not None:

        inputs = get_source_inputs(input_tensor)

    else:

        inputs = img_input

    # Create model.

    model = Model(inputs, x_fc, name='inception_v3')



    # load weights

    if weights == 'imagenet':

        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':

                warnings.warn('You are using the TensorFlow backend, yet you '

                              'are using the Theano '

                              'image data format convention '

                              '(`image_data_format="channels_first"`). '

                              'For best performance, set '

                              '`image_data_format="channels_last"` in '

                              'your Keras config '

                              'at ~/.keras/keras.json.')

        if include_top:

            weights_path = get_file(

                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',

                WEIGHTS_PATH,

                cache_subdir='models',

                md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')

        #else:

        #    weights_path = get_file(

        #        'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',

        #        WEIGHTS_PATH_NO_TOP,

        #        cache_subdir='models',

        #        md5_hash='bcbd6486424b2319ff4ef7d526e38f63')

        model.load_weights(weights_path)

    #

    # define a new version of "bottleneck" with our number of classes

    #

    x_newfc = GlobalAveragePooling2D( name='avg_pool')(x)

    x_newfc = Dense(num_classes, activation='softmax', name='predictions')(x_newfc)



    # Create another model with our customized softmax

    model = Model(inputs, x_newfc)

    # set the first 25 layers 

    # to non-trainable (weights will not be updated)

    #

    # Uncomment the next two lines if you want to trun of tunning of some number of consecutive

    # layers starting from layer 0

    #

    #for layer in model.layers[:25]:

    #    layer.trainable = False

    # Learning rate 0.01 when tunning all layers)

    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])



    return model





def preprocess_input(x):

    x /= 255.

    x -= 0.5

    x *= 2.

    return x

# If you do not have the weights file in your computer Keras will try to connect 

# and download it from:

# https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5

# You can also use your browser to download the file and mv or cp to

#   ~/.keras/models/.



model = InceptionV3(include_top=True, weights='imagenet')



model.summary()
# Start Fine-tuning

nb_train_samples = 9916514

nb_validation_samples = 2454434

epochs = 2

model.fit_generator(

        train_generator,

        steps_per_epoch=nb_train_samples // batch_size,

        epochs=epochs,

        validation_data=validation_generator,

        validation_steps=nb_validation_samples // batch_size)



# you may want to start with steps_per_epoch ~ 100 - 300 and explore the migration to GPU's