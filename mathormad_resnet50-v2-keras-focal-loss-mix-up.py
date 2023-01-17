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
!pip install imgaug --upgrade
import os, sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy

from keras.applications.resnet50 import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score

from keras.utils import Sequence

WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 300

NUM_CLASSES = 1103

beta_f2=2
from keras.legacy import interfaces

from keras.optimizers import Optimizer

from keras import backend as K





class AdamAccumulate_v1(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=20, **kwargs):

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.effective_iterations = K.variable(0, dtype='int64', name='effective_iterations')



            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, dtype='int64')



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)



        self.updates = [K.update(self.iterations, self.iterations + 1)]



        flag = K.equal(self.iterations % self.accum_iters, self.accum_iters - 1)

        flag = K.cast(flag, K.floatx())



        self.updates.append(K.update(self.effective_iterations,

                                     self.effective_iterations + K.cast(flag, 'int64')))



        lr = self.lr

        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * K.cast(self.effective_iterations,

                                                      K.dtype(self.decay))))



        t = K.cast(self.effective_iterations, K.floatx()) + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                     (1. - K.pow(self.beta_1, t)))



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, gg in zip(params, grads, ms, vs, vhats, gs):



            gg_t = (1 - flag) * (gg + g)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / K.cast(self.accum_iters, K.floatx())

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(

                (gg + flag * g) / K.cast(self.accum_iters, K.floatx()))



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - flag * lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, vhat_t))

            else:

                p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append((m, flag * m_t + (1 - flag) * m))

            self.updates.append((v, flag * v_t + (1 - flag) * v))

            self.updates.append((gg, gg_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))





class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):

        if accum_iters < 1:

            raise ValueError('accum_iters must be >= 1')

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))

        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr



        completed_updates = K.cast(K.tf.floor(self.iterations / self.accum_iters), K.floatx())



        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * completed_updates))



        t = completed_updates + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))



        # self.iterations incremented after processing a batch

        # batch:              1 2 3 4 5 6 7 8 9

        # self.iterations:    0 1 2 3 4 5 6 7 8

        # update_switch = 1:        x       x    (if accum_iters=4)

        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)

        update_switch = K.cast(update_switch, K.floatx())



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]



        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):



            sum_grad = tg + g

            avg_grad = sum_grad / self.accum_iters_float



            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))

            else:

                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))

            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))

            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
# Load dataset info

path_to_train = '../input/imet-2019-fgvc6/train/'

data = pd.read_csv('../input/imet-2019-fgvc6/train.csv')



train_dataset_info = []

for name, labels in zip(data['id'], data['attribute_ids'].str.split(' ')):

    train_dataset_info.append({

        'path':os.path.join(path_to_train, name),

        'labels':np.array([int(label) for label in labels])})

train_dataset_info = np.array(train_dataset_info)
gamma = 2.0

epsilon = K.epsilon()

def focal_loss(y_true, y_pred):

    pt = y_pred * y_true + (1-y_pred) * (1-y_true)

    pt = K.clip(pt, epsilon, 1-epsilon)

    CE = -K.log(pt)

    FL = K.pow(1-pt, gamma) * CE

    loss = K.sum(FL, axis=1)

    return loss
sometimes = lambda aug: iaa.Sometimes(0.5, aug)



class data_generator(Sequence):

    

    def mix_up(x, y):

        x = np.array(x, np.float32)

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y

    

    def create_train(dataset_info, batch_size, shape, augument=True, mix=False):

        assert shape[2] == 3

        while True:

            dataset_info = shuffle(dataset_info)

            for start in range(0, len(dataset_info), batch_size):

                end = min(start + batch_size, len(dataset_info))

                batch_images = []

                X_train_batch = dataset_info[start:end]

                batch_labels = np.zeros((len(X_train_batch), NUM_CLASSES))

                for i in range(len(X_train_batch)):

                    image = data_generator.load_image(

                        X_train_batch[i]['path'], augument)

                    batch_images.append(preprocess_input(image))

                    batch_labels[i][X_train_batch[i]['labels']] = 1

                if(mix):

                    batch_images, batch_labels = data_generator.mix_up(batch_images, batch_labels)    

                yield np.array(batch_images, np.float32), batch_labels



    def create_valid(dataset_info, batch_size, shape, augument=False):

        assert shape[2] == 3

        while True:

            # dataset_info = shuffle(dataset_info)

            for start in range(0, len(dataset_info), batch_size):

                end = min(start + batch_size, len(dataset_info))

                batch_images = []

                X_train_batch = dataset_info[start:end]

                batch_labels = np.zeros((len(X_train_batch), NUM_CLASSES))

                for i in range(len(X_train_batch)):

                    image = data_generator.load_image(

                        X_train_batch[i]['path'], augument)

                    batch_images.append(preprocess_input(image))

                    batch_labels[i][X_train_batch[i]['labels']] = 1

                yield np.array(batch_images, np.float32), batch_labels



    def load_image(path, augument=False):

        image = cv2.imread(path+'.png')

        h, w, _ = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if augument:

            image = iaa.CropToFixedSize(height=min(SIZE*2, h),

                            width=min(SIZE*2, w),

                           position='uniform').augment_image(image)

            image = data_generator.augment(image)

        else:

            image = iaa.CropToFixedSize(height=min(SIZE*2, h),

                            width=min(SIZE*2, w),

                           position='center').augment_image(image)

        return cv2.resize(image, (SIZE, SIZE))



    def augment(image):

        augment_img = iaa.Sequential([

        iaa.OneOf([

            iaa.Affine(rotate=0),

            iaa.Affine(rotate=90),

#             iaa.Affine(rotate=180),

#             iaa.Affine(rotate=270)

        ]),

        iaa.OneOf([

            iaa.Fliplr(0.5),

        ])], random_order=True)



        image_aug = augment_img.augment_image(image)

        return image_aug

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model
from keras_applications import imagenet_utils as utils



def ResNet(stack_fn,

           preact,

           use_bias,

           model_name='resnet',

           include_top=True,

           weights='imagenet',

           input_tensor=None,

           input_shape=None,

           pooling=None,

           classes=1000,

           **kwargs):

    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Optionally loads weights pre-trained on ImageNet.

    Note that the data format convention used by the model is

    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments

        stack_fn: a function that returns output tensor for the

            stacked residual blocks.

        preact: whether to use pre-activation or not

            (True for ResNetV2, False for ResNet and ResNeXt).

        use_bias: whether to use biases for convolutional layers or not

            (True for ResNet and ResNetV2, False for ResNeXt).

        model_name: string, model name.

        include_top: whether to include the fully-connected

            layer at the top of the network.

        weights: one of `None` (random initialization),

              'imagenet' (pre-training on ImageNet),

              or the path to the weights file to be loaded.

        input_tensor: optional Keras tensor

            (i.e. output of `layers.Input()`)

            to use as image input for the model.

        input_shape: optional shape tuple, only to be specified

            if `include_top` is False (otherwise the input shape

            has to be `(224, 224, 3)` (with `channels_last` data format)

            or `(3, 224, 224)` (with `channels_first` data format).

            It should have exactly 3 inputs channels.

        pooling: optional pooling mode for feature extraction

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

    # Returns

        A Keras model instance.

    # Raises

        ValueError: in case of invalid argument for `weights`,

            or invalid input shape.

    """

    global backend, layers, models, keras_utils

    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    backend, layers, models, keras_utils = keras.backend, keras.layers, keras.models, keras.utils



    if not (weights in {'imagenet', None} or os.path.exists(weights)):

        raise ValueError('The `weights` argument should be either '

                         '`None` (random initialization), `imagenet` '

                         '(pre-training on ImageNet), '

                         'or the path to the weights file to be loaded.')



    if weights == 'imagenet' and include_top and classes != 1000:

        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'

                         ' as true, `classes` should be 1000')



    # Determine proper input shape

    input_shape = utils._obtain_input_shape(input_shape,

                                          default_size=224,

                                          min_size=32,

                                          data_format=backend.image_data_format(),

                                          require_flatten=include_top,

                                          weights=weights)



    if input_tensor is None:

        img_input = layers.Input(shape=input_shape)

    else:

        if not backend.is_keras_tensor(input_tensor):

            img_input = layers.Input(tensor=input_tensor, shape=input_shape)

        else:

            img_input = input_tensor



    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1



    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)

    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)



    if preact is False:

        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                      name='conv1_bn')(x)

        x = layers.Activation('relu', name='conv1_relu')(x)



    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)

    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)



    x = stack_fn(x)



    if preact is True:

        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                      name='post_bn')(x)

        x = layers.Activation('relu', name='post_relu')(x)



    if include_top:

        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

        x = layers.Dense(classes, activation='softmax', name='probs')(x)

    else:

        if pooling == 'avg':

            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

        elif pooling == 'max':

            x = layers.GlobalMaxPooling2D(name='max_pool')(x)



    # Ensure that the model takes into account

    # any potential predecessors of `input_tensor`.

    if input_tensor is not None:

        inputs = keras_utils.get_source_inputs(input_tensor)

    else:

        inputs = img_input



    # Create model.

    model = models.Model(inputs, x, name=model_name)



    # Load weights.

    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):

        if include_top:

            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'

            file_hash = WEIGHTS_HASHES[model_name][0]

        else:

            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'

            file_hash = WEIGHTS_HASHES[model_name][1]

        weights_path = keras_utils.get_file(file_name,

                                            BASE_WEIGHTS_PATH + file_name,

                                            cache_subdir='models',

                                            file_hash=file_hash)

        model.load_weights(weights_path)

    elif weights is not None:

        model.load_weights(weights)



    return model
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



def block2(x, filters, kernel_size=3, stride=1,

           conv_shortcut=False, name=None):

    """A residual block.

    # Arguments

        x: input tensor.

        filters: integer, filters of the bottleneck layer.

        kernel_size: default 3, kernel size of the bottleneck layer.

        stride: default 1, stride of the first layer.

        conv_shortcut: default False, use convolution shortcut if True,

            otherwise identity shortcut.

        name: string, block label.

    # Returns

        Output tensor for the residual block.

    """

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1



    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                       name=name + '_preact_bn')(x)

    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)



    if conv_shortcut is True:

        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,

                                 name=name + '_0_conv')(preact)

    else:

        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x



    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,

                      name=name + '_1_conv')(preact)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                  name=name + '_1_bn')(x)

    x = layers.Activation('relu', name=name + '_1_relu')(x)



    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)

    x = layers.Conv2D(filters, kernel_size, strides=stride,

                      use_bias=False, name=name + '_2_conv')(x)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,

                                  name=name + '_2_bn')(x)

    x = layers.Activation('relu', name=name + '_2_relu')(x)



    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)

    x = layers.Add(name=name + '_out')([shortcut, x])

    return x





def stack2(x, filters, blocks, stride1=2, name=None):

    """A set of stacked residual blocks.

    # Arguments

        x: input tensor.

        filters: integer, filters of the bottleneck layer in a block.

        blocks: integer, blocks in the stacked blocks.

        stride1: default 2, stride of the first layer in the first block.

        name: string, stack label.

    # Returns

        Output tensor for the stacked blocks.

    """

    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')

    for i in range(2, blocks):

        x = block2(x, filters, name=name + '_block' + str(i))

    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))

    return x



def ResNet50V2(include_top=True,

               weights='imagenet',

               input_tensor=None,

               input_shape=None,

               pooling=None,

               classes=1000,

               **kwargs):

    def stack_fn(x):

        x = stack2(x, 64, 3, name='conv2')

        x = stack2(x, 128, 4, name='conv3')

        x = stack2(x, 256, 6, name='conv4')

        x = stack2(x, 512, 3, stride1=1, name='conv5')

        return x

    return ResNet(stack_fn, True, True, 'resnet50v2',

                  include_top, weights,

                  input_tensor, input_shape,

                  pooling, classes,

                  **kwargs)
# reference link: https://gist.github.com/drscotthawley/d1818aabce8d1bf082a6fb37137473ae

from keras.callbacks import Callback



def get_1cycle_schedule(lr_max=1e-3, n_data_points=8000, epochs=200, batch_size=40, verbose=0):          

    """

    Creates a look-up table of learning rates for 1cycle schedule with cosine annealing

    See @sgugger's & @jeremyhoward's code in fastai library: https://github.com/fastai/fastai/blob/master/fastai/train.py

    Wrote this to use with my Keras and (non-fastai-)PyTorch codes.

    Note that in Keras, the LearningRateScheduler callback (https://keras.io/callbacks/#learningratescheduler) only operates once per epoch, not per batch

      So see below for Keras callback



    Keyword arguments:

    lr_max            chosen by user after lr_finder

    n_data_points     data points per epoch (e.g. size of training set)

    epochs            number of epochs

    batch_size        batch size

    Output:  

    lrs               look-up table of LR's, with length equal to total # of iterations

    Then you can use this in your PyTorch code by counting iteration number and setting

          optimizer.param_groups[0]['lr'] = lrs[iter_count]

    """

    if verbose > 0:

        print("Setting up 1Cycle LR schedule...")

    pct_start, div_factor = 0.3, 25.        # @sgugger's parameters in fastai code

    lr_start = lr_max/div_factor

    lr_end = lr_start/1e4

    n_iter = (n_data_points * epochs // batch_size) + 1    # number of iterations

    a1 = int(n_iter * pct_start)

    a2 = n_iter - a1



    # make look-up table

    lrs_first = np.linspace(lr_start, lr_max, a1)            # linear growth

    lrs_second = (lr_max-lr_end)*(1+np.cos(np.linspace(0,np.pi,a2)))/2 + lr_end  # cosine annealing

    lrs = np.concatenate((lrs_first, lrs_second))

    return lrs





class OneCycleScheduler(Callback):

    """My modification of Keras' Learning rate scheduler to do 1Cycle learning

       which increments per BATCH, not per epoch

    Keyword arguments

        **kwargs:  keyword arguments to pass to get_1cycle_schedule()

        Also, verbose: int. 0: quiet, 1: update messages.



    Sample usage (from my train.py):

        lrsched = OneCycleScheduler(lr_max=1e-4, n_data_points=X_train.shape[0],

        epochs=epochs, batch_size=batch_size, verbose=1)

    """

    def __init__(self, **kwargs):

        super(OneCycleScheduler, self).__init__()

        self.verbose = kwargs.get('verbose', 0)

        self.lrs = get_1cycle_schedule(**kwargs)

        self.iteration = 0



    def on_batch_begin(self, batch, logs=None):

        lr = self.lrs[self.iteration]

        K.set_value(self.model.optimizer.lr, lr)         # here's where the assignment takes place

        if self.verbose > 0:

            print('\nIteration %06d: OneCycleScheduler setting learning '

                  'rate to %s.' % (self.iteration, lr))

        self.iteration += 1



    def on_epoch_end(self, epoch, logs=None):  # this is unchanged from Keras LearningRateScheduler

        logs = logs or {}

        logs['lr'] = K.get_value(self.model.optimizer.lr)

        # self.iteration = 0



from keras.regularizers import l2



def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = ResNet50V2(include_top=False,

                   weights=None,

                   input_tensor=input_tensor)

    base_model.load_weights('../input/keras-pretrain-model-weights/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5')

#     x = Conv2D(32, kernel_size=(1,1), activation='relu')(base_model.output)

#     x = Flatten()(x)

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.3)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.3)(x)

#     x = Dense(1024, activation='relu')(x)

#     x = Dropout(0.3)(x)

    final_output = Dense(n_out, activation='sigmoid', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)

                             

from sklearn.model_selection import train_test_split



epochs = 20; batch_size = 64

checkpoint = ModelCheckpoint('../working/Resnet50_focal.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 

                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=7)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)





# split data into train, valid

indexes = np.arange(train_dataset_info.shape[0])

train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=8)



# create train and valid datagens

train_generator = data_generator.create_train(

    train_dataset_info[train_indexes], batch_size, (SIZE,SIZE,3), augument=True, mix=False)

train_mixup = data_generator.create_train(

    train_dataset_info[train_indexes], batch_size, (SIZE,SIZE,3), augument=True, mix=True)

train_generator_warmup = data_generator.create_train(

    train_dataset_info[train_indexes], batch_size, (SIZE,SIZE,3), augument=False)

validation_generator = data_generator.create_valid(

    train_dataset_info[valid_indexes], batch_size, (SIZE,SIZE,3), augument=False)



lrsched = OneCycleScheduler(lr_max=1e-3, n_data_points=len(train_indexes),

        epochs=epochs, batch_size=batch_size, verbose=0)

# callbacks_list = [checkpoint, csv_logger, lrsched]

callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]

# callbacks_list = [checkpoint, csv_logger]
import matplotlib.pyplot as plt

    

lrs = get_1cycle_schedule(lr_max=1e-3, n_data_points=len(train_indexes), epochs=20, batch_size=64, verbose=0)

epoch_list = np.linspace(0,epochs,len(lrs))

plt.plot(epoch_list,lrs)
model = create_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=NUM_CLASSES)
# warm up model

for layer in model.layers:

    layer.trainable = False



for i in range(-3,0):

    model.layers[i].trainable = True



model.compile(

    loss='binary_crossentropy',

    optimizer=Adam(1e-3))

    # optimizer=AdamAccumulate(lr=1e-3, accum_iters=2))



model.fit_generator(

    train_generator_warmup,

    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),

    epochs=2,

    max_queue_size=16, workers=WORKERS, use_multiprocessing=True,

    verbose=1)
# train all layers

for layer in model.layers:

    layer.trainable = True



callbacks_list = [checkpoint, csv_logger, lrsched]

model.compile(loss='binary_crossentropy',

            # loss=focal_loss,

            optimizer=Adam(lr=1e-3,decay=3e-4))

            # optimizer=AdamAccumulate(lr=2e-4, accum_iters=2))



model.fit_generator(

    # train_generator,

    train_mixup,

    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),

    validation_data=validation_generator,

    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),

    epochs=epochs,

    verbose=1,

    max_queue_size=16, workers=WORKERS, use_multiprocessing=True,

    callbacks=callbacks_list)
# model.compile(loss='binary_crossentropy',

#             # loss=focal_loss,

#             optimizer=Adam(lr=1e-4))

#             # optimizer=AdamAccumulate(lr=1e-4, accum_iters=4))



# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]



# model.fit_generator(

#     train_generator,

#     steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),

#     validation_data=validation_generator,

#     validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),

#     epochs=epochs,

#     verbose=1,

#     max_queue_size=16, workers=WORKERS, use_multiprocessing=True,

#     callbacks=callbacks_list)
print(os.listdir('../working/'))
submit = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')

model.load_weights('../working/Resnet50_focal.h5')

predicted = []
'''Search for the best threshold regarding the validation set'''



BATCH = 512

fullValGen = data_generator.create_valid(

    train_dataset_info[valid_indexes], BATCH, (SIZE,SIZE,3))



n_val = round(train_dataset_info.shape[0]*0.15)//BATCH

print(n_val)



lastFullValPred = np.empty((0, NUM_CLASSES))

lastFullValLabels = np.empty((0, NUM_CLASSES))

for i in tqdm(range(n_val+1)): 

    im, lbl = next(fullValGen)

    scores = model.predict(im)

    lastFullValPred = np.append(lastFullValPred, scores, axis=0)

    lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)

print(lastFullValPred.shape, lastFullValLabels.shape)
def my_f2(y_true, y_pred):

    assert y_true.shape[0] == y_pred.shape[0]



    tp = np.sum((y_true == 1) & (y_pred == 1), axis=1)

    tn = np.sum((y_true == 0) & (y_pred == 0), axis=1)

    fp = np.sum((y_true == 0) & (y_pred == 1), axis=1)

    fn = np.sum((y_true == 1) & (y_pred == 0), axis=1)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f2 = (1+beta_f2**2)*p*r / (p*beta_f2**2 + r + 1e-15)



    return np.mean(f2)



def find_best_fixed_threshold(preds, targs, do_plot=True):

    score = []

    thrs = np.arange(0, 0.5, 0.01)

    for thr in tqdm(thrs):

        score.append(my_f2(targs, (preds > thr).astype(int) ))

    score = np.array(score)

    pm = score.argmax()

    best_thr, best_score = thrs[pm], score[pm].item()

    print(f'thr={best_thr:.3f}', f'F2={best_score:.3f}')

    if do_plot:

        plt.plot(thrs, score)

        plt.vlines(x=best_thr, ymin=score.min(), ymax=score.max())

        plt.text(best_thr+0.03, best_score-0.01, f'$F_{2}=${best_score:.3f}', fontsize=14);

        plt.show()

    return best_thr, best_score
best_thr, best_score = find_best_fixed_threshold(lastFullValPred, lastFullValLabels, do_plot=True)
# for i, name in tqdm(enumerate(submit['id'])):

#     path = os.path.join('../input/imet-2019-fgvc6/test/', name)

#     image = data_generator.load_image(path, (SIZE,SIZE,3))

#     score_predict = model.predict(preprocess_input(image[np.newaxis]))

#     # score_predict = model.predict((image[np.newaxis])/255)

#     # print(score_predict)

#     label_predict = np.arange(NUM_CLASSES)[score_predict[0]>=best_thr]

#     # print(label_predict)

#     str_predict_label = ' '.join(str(l) for l in label_predict)

#     predicted.append(str_predict_label)
# submit['attribute_ids'] = predicted

# submit.to_csv('submission.csv', index=False)