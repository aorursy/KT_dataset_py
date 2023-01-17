# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D



import os

import gc

import time

from IPython.display import clear_output

from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint as MC

from tensorflow.keras import backend as K

import pydicom

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Reading test data...')

test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")

print(test.shape)

test.head()
from keras import regularizers

REG = 1e-4

DO = 0

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================

# pylint: disable=invalid-name

"""EfficientNet models for Keras.



Reference paper:

  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]

    (https://arxiv.org/abs/1905.11946) (ICML 2019)

"""

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import copy

import math

import os



from tensorflow.python.keras import backend

from tensorflow.python.keras import layers

from tensorflow.python.keras.applications import imagenet_utils

from tensorflow.python.keras.engine import training

from tensorflow.python.keras.utils import data_utils

from tensorflow.python.keras.utils import layer_utils

from tensorflow.python.util.tf_export import keras_export





BASE_WEIGHTS_PATH = 'https://storage.googleapis.com/keras-applications/'



WEIGHTS_HASHES = {

    'b0': ('902e53a9f72be733fc0bcb005b3ebbac',

           '50bc09e76180e00e4465e1a485ddc09d'),

    'b1': ('1d254153d4ab51201f1646940f018540',

           '74c4e6b3e1f6a1eea24c589628592432'),

    'b2': ('b15cce36ff4dcbd00b6dd88e7857a6ad',

           '111f8e2ac8aa800a7a99e3239f7bfb39'),

    'b3': ('ffd1fdc53d0ce67064dc6a9c7960ede0',

           'af6d107764bb5b1abb91932881670226'),

    'b4': ('18c95ad55216b8f92d7e70b3a046e2fc',

           'ebc24e6d6c33eaebbd558eafbeedf1ba'),

    'b5': ('ace28f2a6363774853a83a0b21b9421a',

           '38879255a25d3c92d5e44e04ae6cec6f'),

    'b6': ('165f6e37dce68623721b423839de8be5',

           '9ecce42647a20130c1f39a5d4cb75743'),

    'b7': ('8c03f828fec3ef71311cd463b6759d99',

           'cbcfe4450ddf6f3ad90b1b398090fe4a'),

}



DEFAULT_BLOCKS_ARGS = [{

    'kernel_size': 3,

    'repeats': 1,

    'filters_in': 32,

    'filters_out': 16,

    'expand_ratio': 1,

    'id_skip': True,

    'strides': 1,

    'se_ratio': 0.25

}, {

    'kernel_size': 3,

    'repeats': 2,

    'filters_in': 16,

    'filters_out': 24,

    'expand_ratio': 6,

    'id_skip': True,

    'strides': 2,

    'se_ratio': 0.25

}, {

    'kernel_size': 5,

    'repeats': 2,

    'filters_in': 24,

    'filters_out': 40,

    'expand_ratio': 6,

    'id_skip': True,

    'strides': 2,

    'se_ratio': 0.25

}, {

    'kernel_size': 3,

    'repeats': 3,

    'filters_in': 40,

    'filters_out': 80,

    'expand_ratio': 6,

    'id_skip': True,

    'strides': 2,

    'se_ratio': 0.25

}, {

    'kernel_size': 5,

    'repeats': 3,

    'filters_in': 80,

    'filters_out': 112,

    'expand_ratio': 6,

    'id_skip': True,

    'strides': 1,

    'se_ratio': 0.25

}, {

    'kernel_size': 5,

    'repeats': 4,

    'filters_in': 112,

    'filters_out': 192,

    'expand_ratio': 6,

    'id_skip': True,

    'strides': 2,

    'se_ratio': 0.25

}, {

    'kernel_size': 3,

    'repeats': 1,

    'filters_in': 192,

    'filters_out': 320,

    'expand_ratio': 6,

    'id_skip': True,

    'strides': 1,

    'se_ratio': 0.25

}]



CONV_KERNEL_INITIALIZER = {

    'class_name': 'VarianceScaling',

    'config': {

        'scale': 2.0,

        'mode': 'fan_out',

        'distribution': 'truncated_normal'

    }

}



DENSE_KERNEL_INITIALIZER = {

    'class_name': 'VarianceScaling',

    'config': {

        'scale': 1. / 3.,

        'mode': 'fan_out',

        'distribution': 'uniform'

    }

}





def EfficientNet(

    width_coefficient,

    depth_coefficient,

    default_size,

    dropout_rate=0.2,

    drop_connect_rate=0.2,

    depth_divisor=8,

    activation='swish',

    blocks_args='default',

    model_name='efficientnet',

    include_top=True,

    weights='imagenet',

    input_tensor=None,

    input_shape=None,

    pooling=None,

    classes=1000,

    classifier_activation='softmax',

):

  """Instantiates the EfficientNet architecture using given scaling coefficients.



  Optionally loads weights pre-trained on ImageNet.

  Note that the data format convention used by the model is

  the one specified in your Keras config at `~/.keras/keras.json`.



  Arguments:

    width_coefficient: float, scaling coefficient for network width.

    depth_coefficient: float, scaling coefficient for network depth.

    default_size: integer, default input image size.

    dropout_rate: float, dropout rate before final classifier layer.

    drop_connect_rate: float, dropout rate at skip connections.

    depth_divisor: integer, a unit of network width.

    activation: activation function.

    blocks_args: list of dicts, parameters to construct block modules.

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

        if `include_top` is False.

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

    classifier_activation: A `str` or callable. The activation function to use

        on the "top" layer. Ignored unless `include_top=True`. Set

        `classifier_activation=None` to return the logits of the "top" layer.



  Returns:

    A `keras.Model` instance.



  Raises:

    ValueError: in case of invalid argument for `weights`,

      or invalid input shape.

    ValueError: if `classifier_activation` is not `softmax` or `None` when

      using a pretrained top layer.

  """

  if blocks_args == 'default':

    blocks_args = DEFAULT_BLOCKS_ARGS



  if not (weights in {'imagenet', None} or os.path.exists(weights)):

    raise ValueError('The `weights` argument should be either '

                     '`None` (random initialization), `imagenet` '

                     '(pre-training on ImageNet), '

                     'or the path to the weights file to be loaded.')



  if weights == 'imagenet' and include_top and classes != 1000:

    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'

                     ' as true, `classes` should be 1000')



  # Determine proper input shape

  input_shape = imagenet_utils.obtain_input_shape(

      input_shape,

      default_size=default_size,

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



  def round_filters(filters, divisor=depth_divisor):

    """Round number of filters based on depth multiplier."""

    filters *= width_coefficient

    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.

    if new_filters < 0.9 * filters:

      new_filters += divisor

    return int(new_filters)



  def round_repeats(repeats):

    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))



  # Build stem

  x = img_input

  #x = layers.Rescaling(1. / 255.)(x)

  x = layers.Normalization(axis=bn_axis)(x)



  x = layers.ZeroPadding2D(

      padding=imagenet_utils.correct_pad(x, 3),

      name='stem_conv_pad')(x)

  x = layers.Conv2D(

      round_filters(32),

      3,

      strides=2,

      padding='valid',

      use_bias=False,

      kernel_initializer=CONV_KERNEL_INITIALIZER,

      name='stem_conv',

      kernel_regularizer=regularizers.l2(REG))(x)

  x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)

  x = layers.Activation(activation, name='stem_activation')(x)

  #x = layers.Dropout(0.01, name='top_dropout0')(x)



  # Build blocks

  blocks_args = copy.deepcopy(blocks_args)



  b = 0

  blocks = float(sum(args['repeats'] for args in blocks_args))

  for (i, args) in enumerate(blocks_args):

    assert args['repeats'] > 0

    # Update block input and output filters based on depth multiplier.

    args['filters_in'] = round_filters(args['filters_in'])

    args['filters_out'] = round_filters(args['filters_out'])



    for j in range(round_repeats(args.pop('repeats'))):

      # The first block needs to take care of stride and filter size increase.

      if j > 0:

        args['strides'] = 1

        args['filters_in'] = args['filters_out']

      x = block(

          x,

          activation,

          drop_connect_rate * b / blocks,

          name='block{}{}_'.format(i + 1, chr(j + 97)),

          **args)

      b += 1



  # Build top

  x = layers.Conv2D(

      round_filters(1280),

      1,

      padding='same',

      use_bias=False,

      kernel_initializer=CONV_KERNEL_INITIALIZER,

      name='top_conv',

      kernel_regularizer=regularizers.l2(REG))(x)

  x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)

  x = layers.Activation(activation, name='top_activation')(x)

  if include_top:

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if dropout_rate > 0:

      x = layers.Dropout(dropout_rate, name='top_dropout1')(x)

    imagenet_utils.validate_activation(classifier_activation, weights)

    x = layers.Dense(

        classes,

        activation=classifier_activation,

        kernel_initializer=DENSE_KERNEL_INITIALIZER,

        name='predictions',

          kernel_regularizer=regularizers.l2(REG))(x)

  else:

    if pooling == 'avg':

      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    elif pooling == 'max':

      x = layers.GlobalMaxPooling2D(name='max_pool')(x)



  # Ensure that the model takes into account

  # any potential predecessors of `input_tensor`.

  if input_tensor is not None:

    inputs = layer_utils.get_source_inputs(input_tensor)

  else:

    inputs = img_input



  # Create model.

  model = training.Model(inputs, x, name=model_name)



  # Load weights.

  if weights == 'imagenet':

    if include_top:

      file_suffix = '.h5'

      file_hash = WEIGHTS_HASHES[model_name[-2:]][0]

    else:

      file_suffix = '_notop.h5'

      file_hash = WEIGHTS_HASHES[model_name[-2:]][1]

    file_name = model_name + file_suffix

    weights_path = data_utils.get_file(

        file_name,

        BASE_WEIGHTS_PATH + file_name,

        cache_subdir='models',

        file_hash=file_hash)

    model.load_weights(weights_path)

  elif weights is not None:

    model.load_weights(weights)

  return model





def block(inputs,

          activation='swish',

          drop_rate=0.,

          name='',

          filters_in=32,

          filters_out=16,

          kernel_size=3,

          strides=1,

          expand_ratio=1,

          se_ratio=0.,

          id_skip=True):

  """An inverted residual block.



  Arguments:

      inputs: input tensor.

      activation: activation function.

      drop_rate: float between 0 and 1, fraction of the input units to drop.

      name: string, block label.

      filters_in: integer, the number of input filters.

      filters_out: integer, the number of output filters.

      kernel_size: integer, the dimension of the convolution window.

      strides: integer, the stride of the convolution.

      expand_ratio: integer, scaling coefficient for the input filters.

      se_ratio: float between 0 and 1, fraction to squeeze the input filters.

      id_skip: boolean.



  Returns:

      output tensor for the block.

  """

  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1



  # Expansion phase

  filters = filters_in * expand_ratio

  if expand_ratio != 1:

    x = layers.Conv2D(

        filters,

        1,

        padding='same',

        use_bias=False,

        kernel_initializer=CONV_KERNEL_INITIALIZER,

        name=name + 'expand_conv',

          kernel_regularizer=regularizers.l2(REG))(

            inputs)

    x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)

    x = layers.Activation(activation, name=name + 'expand_activation')(x)

    #x = layers.Dropout(0.01, name=name+'top_dropout3')(x)

  else:

    x = inputs



  # Depthwise Convolution

  if strides == 2:

    x = layers.ZeroPadding2D(

        padding=imagenet_utils.correct_pad(x, kernel_size),

        name=name + 'dwconv_pad')(x)

    conv_pad = 'valid'

  else:

    conv_pad = 'same'

  x = layers.DepthwiseConv2D(

      kernel_size,

      strides=strides,

      padding=conv_pad,

      use_bias=False,

      depthwise_initializer=CONV_KERNEL_INITIALIZER,

      name=name + 'dwconv')(x)

  x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)

  x = layers.Activation(activation, name=name + 'activation')(x)

  #x = layers.Dropout(0.01, name=name+'top_dropout')(x)



  # Squeeze and Excitation phase

  if 0 < se_ratio <= 1:

    filters_se = max(1, int(filters_in * se_ratio))

    se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)

    se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)

    se = layers.Conv2D(

        filters_se,

        1,

        padding='same',

        activation=activation,

        kernel_initializer=CONV_KERNEL_INITIALIZER,

        name=name + 'se_reduce',

          kernel_regularizer=regularizers.l2(REG))(

            se)

    #se = layers.Dropout(0.1, name=name+'top_dropout5')(se)

    se = layers.Conv2D(

        filters,

        1,

        padding='same',

        activation='sigmoid',

        kernel_initializer=CONV_KERNEL_INITIALIZER,

        name=name + 'se_expand',

          kernel_regularizer=regularizers.l2(REG))(se)

    #se = layers.Dropout(0.1, name=name+'top_dropout6')(se)

    x = layers.multiply([x, se], name=name + 'se_excite')



  # Output phase

  x = layers.Conv2D(

      filters_out,

      1,

      padding='same',

      use_bias=False,

      kernel_initializer=CONV_KERNEL_INITIALIZER,

      name=name + 'project_conv',

      kernel_regularizer=regularizers.l2(REG))(x)

  x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)

  if id_skip and strides == 1 and filters_in == filters_out:

    if drop_rate > 0:

      x = layers.Dropout(

          drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)

    x = layers.add([x, inputs], name=name + 'add')

  return x





@keras_export('keras.applications.efficientnet.EfficientNetB0',

              'keras.applications.EfficientNetB0')

def EfficientNetB0(include_top=True,

                   weights='imagenet',

                   input_tensor=None,

                   input_shape=None,

                   pooling=None,

                   classes=1000,

                   **kwargs):

  return EfficientNet(

      1.0,

      1.0,

      224,

      0.5,

      model_name='efficientnetb0',

      include_top=include_top,

      weights=weights,

      input_tensor=input_tensor,

      input_shape=input_shape,

      pooling=pooling,

      classes=classes,

      **kwargs)





@keras_export('keras.applications.efficientnet.EfficientNetB1',

              'keras.applications.EfficientNetB1')

def EfficientNetB1(include_top=True,

                   weights='imagenet',

                   input_tensor=None,

                   input_shape=None,

                   pooling=None,

                   classes=1000,

                   **kwargs):

  return EfficientNet(

      1.0,

      1.1,

      240,

      0.2,

      model_name='efficientnetb1',

      include_top=include_top,

      weights=weights,

      input_tensor=input_tensor,

      input_shape=input_shape,

      pooling=pooling,

      classes=classes,

      **kwargs)





@keras_export('keras.applications.efficientnet.EfficientNetB2',

              'keras.applications.EfficientNetB2')

def EfficientNetB2(include_top=True,

                   weights='imagenet',

                   input_tensor=None,

                   input_shape=None,

                   pooling=None,

                   classes=1000,

                   **kwargs):

  return EfficientNet(

      1.1,

      1.2,

      260,

      0.3,

      model_name='efficientnetb2',

      include_top=include_top,

      weights=weights,

      input_tensor=input_tensor,

      input_shape=input_shape,

      pooling=pooling,

      classes=classes,

      **kwargs)





@keras_export('keras.applications.efficientnet.EfficientNetB3',

              'keras.applications.EfficientNetB3')

def EfficientNetB3(include_top=True,

                   weights='imagenet',

                   input_tensor=None,

                   input_shape=None,

                   pooling=None,

                   classes=1000,

                   **kwargs):

  return EfficientNet(

      1.2,

      1.4,

      300,

      0.3,

      model_name='efficientnetb3',

      include_top=include_top,

      weights=weights,

      input_tensor=input_tensor,

      input_shape=input_shape,

      pooling=pooling,

      classes=classes,

      **kwargs)





@keras_export('keras.applications.efficientnet.EfficientNetB4',

              'keras.applications.EfficientNetB4')

def EfficientNetB4(include_top=True,

                   weights='imagenet',

                   input_tensor=None,

                   input_shape=None,

                   pooling=None,

                   classes=1000,

                   **kwargs):

  return EfficientNet(

      1.4,

      1.8,

      380,

      0.4,

      model_name='efficientnetb4',

      include_top=include_top,

      weights=weights,

      input_tensor=input_tensor,

      input_shape=input_shape,

      pooling=pooling,

      classes=classes,

      **kwargs)





@keras_export('keras.applications.efficientnet.EfficientNetB5',

              'keras.applications.EfficientNetB5')

def EfficientNetB5(include_top=True,

                   weights='imagenet',

                   input_tensor=None,

                   input_shape=None,

                   pooling=None,

                   classes=1000,

                   **kwargs):

  return EfficientNet(

      1.6,

      2.2,

      456,

      0.4,

      model_name='efficientnetb5',

      include_top=include_top,

      weights=weights,

      input_tensor=input_tensor,

      input_shape=input_shape,

      pooling=pooling,

      classes=classes,

      **kwargs)





@keras_export('keras.applications.efficientnet.EfficientNetB6',

              'keras.applications.EfficientNetB6')

def EfficientNetB6(include_top=True,

                   weights='imagenet',

                   input_tensor=None,

                   input_shape=None,

                   pooling=None,

                   classes=1000,

                   **kwargs):

  return EfficientNet(

      1.8,

      2.6,

      528,

      0.5,

      model_name='efficientnetb6',

      include_top=include_top,

      weights=weights,

      input_tensor=input_tensor,

      input_shape=input_shape,

      pooling=pooling,

      classes=classes,

      **kwargs)





@keras_export('keras.applications.efficientnet.EfficientNetB7',

              'keras.applications.EfficientNetB7')

def EfficientNetB7(include_top=True,

                   weights='imagenet',

                   input_tensor=None,

                   input_shape=None,

                   pooling=None,

                   classes=1000,

                   **kwargs):

  return EfficientNet(

      2.0,

      3.1,

      600,

      0.5,

      model_name='efficientnetb7',

      include_top=include_top,

      weights=weights,

      input_tensor=input_tensor,

      input_shape=input_shape,

      pooling=pooling,

      classes=classes,

      **kwargs)





@keras_export('keras.applications.efficientnet.preprocess_input')

def preprocess_input(x, data_format=None):  # pylint: disable=unused-argument

  return x





@keras_export('keras.applications.efficientnet.decode_predictions')

def decode_predictions(preds, top=5):

  """Decodes the prediction result from the model.



  Arguments

    preds: Numpy tensor encoding a batch of predictions.

    top: Integer, how many top-guesses to return.



  Returns

    A list of lists of top class prediction tuples

    `(class_name, class_description, score)`.

    One list of tuples per sample in batch input.



  Raises

    ValueError: In case of invalid shape of the `preds` array (must be 2D).

  """

  return imagenet_utils.decode_predictions(preds, top=top)
def build_model(train_type=0):

    inputs = Input((256, 256, 3))

    #x = Conv2D(3, (1, 1), activation='relu')(inputs)

    base_model = EfficientNetB3(

        include_top=False,

        weights=None,

        input_shape=[256,256,3]

    )

    #print(len(base_model.layers))

    if train_type==1:

        base_model.trainable = False

    

    if train_type==2:

        for layer in base_model.layers[-20:]:

            if not isinstance(layer, layers.BatchNormalization):

                layer.trainable = True



    outputs = base_model(inputs)#, training=True)

    outputs = keras.layers.GlobalAveragePooling2D()(outputs)

    outputs = layers.BatchNormalization()(outputs)

    outputs = Dropout(0.25)(outputs)

    nefp = Dense(1, activation='sigmoid', name='negative_exam_for_pe')(outputs)

    rlrg1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_gte_1')(outputs)

    rlrl1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_lt_1')(outputs) 

    lspe = Dense(1, activation='sigmoid', name='leftsided_pe')(outputs)

    cpe = Dense(1, activation='sigmoid', name='chronic_pe')(outputs)

    rspe = Dense(1, activation='sigmoid', name='rightsided_pe')(outputs)

    aacpe = Dense(1, activation='sigmoid', name='acute_and_chronic_pe')(outputs)

    cnpe = Dense(1, activation='sigmoid', name='central_pe')(outputs)

    indt = Dense(1, activation='sigmoid', name='indeterminate')(outputs)



    model = Model(inputs=inputs, outputs={'negative_exam_for_pe':nefp,

                                          'rv_lv_ratio_gte_1':rlrg1,

                                          'rv_lv_ratio_lt_1':rlrl1,

                                          'leftsided_pe':lspe,

                                          'chronic_pe':cpe,

                                          'rightsided_pe':rspe,

                                          'acute_and_chronic_pe':aacpe,

                                          'central_pe':cnpe,

                                          'indeterminate':indt})



    opt = keras.optimizers.Adam(lr=0.001)

    #loss = binary_focal_loss()

    model.compile(optimizer=opt,

                  #loss=loss,

                  loss='binary_crossentropy',

                  metrics=['AUC'])

    return model
def convert_to_rgb(array):

    array = array.reshape((256, 256, 3))

    return array#np.stack([array, array, array], axis=2).reshape((256, 256, 3))

    

def custom_dcom_image_generator(batch_size, dataset, test=False, debug=False):

    

    fnames = dataset[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']]

    

    if not test:

        Y = dataset[['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',

                     'leftsided_pe', 'chronic_pe', 'rightsided_pe',

                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']]

        prefix = 'input/rsna-str-pulmonary-embolism-detection/train'

        

    else:

        prefix = 'input/rsna-str-pulmonary-embolism-detection/test'

    

    X = []

    batch = 0

    for st, sr, so in fnames.values:

        if debug:

            print(f"Current file: ../{prefix}/{st}/{sr}/{so}.dcm")



        dicom = get_img(f"../{prefix}/{st}/{sr}/{so}.dcm")

        image = convert_to_rgb(dicom)

        X.append(image)

        

        del st, sr, so

        

        if len(X) == batch_size:

            if test:

                yield np.array(X)

                del X

            else:

                yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values

                del X

                

            gc.collect()

            X = []

            batch += 1

        

    if test:

        yield np.array(X)

    else:

        yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values

        del Y

    del X

    gc.collect()

    return
MAX_LENGTH = 256.

from scipy.ndimage.interpolation import zoom

def window(img, WL=50, WW=350):

    upper, lower = WL+WW//2, WL-WW//2

    X = np.clip(img.copy(), lower, upper)

    X = X - np.min(X)

    X = X / np.max(X)

    #X = (X*255.0)

    return X



def img_convert(image):

    image_lung = np.expand_dims(window(image, WL=-600, WW=1500), axis=-1)

    image_mediastinal = np.expand_dims(window(image, WL=40, WW=400), axis=-1)

    image_pe_specific = np.expand_dims(window(image, WL=100, WW=700), axis=-1)

    image = np.concatenate([image_mediastinal, image_pe_specific, image_lung], axis=-1)

    rat = MAX_LENGTH / np.max(image.shape[1:])

    image = zoom(image, [rat,rat,1.], prefilter=False, order=1)

    return image
cnt=0
import vtk

from vtk.util import numpy_support

import cv2



reader = vtk.vtkDICOMImageReader()





def get_img(path):

    dicoms = pydicom.dcmread(path)

    M = float(dicoms.RescaleSlope)

    B = float(dicoms.RescaleIntercept)

    # Assume all images are axial

    z_pos = [float(dicoms.ImagePositionPatient[-1])]

    dicoms = np.asarray([dicoms.pixel_array])

    dicoms = dicoms[np.argsort(z_pos)]

    dicoms = dicoms * M

    dicoms = dicoms + B

    image = dicoms

    image_lung = np.expand_dims(window(image, WL=-600, WW=1500), axis=3)

    image_mediastinal = np.expand_dims(window(image, WL=40, WW=400), axis=3)

    image_pe_specific = np.expand_dims(window(image, WL=100, WW=700), axis=3)

    image = np.concatenate([image_mediastinal, image_pe_specific, image_lung], axis=3)

    rat = MAX_LENGTH / np.max(image.shape[1:])

    image = zoom(image, [1.,rat,rat,1.], prefilter=False, order=1)



    #plt.imshow(image[0])

    #plt.show()



    return image[0]



from tensorflow.keras import backend as K



predictions = {}

stopper = 3600 * 8.75 #9 hours limit for prediction

pred_start_time = time.time()



p, c = time.time(), time.time()

batch_size = 1536

    

l = 0

n = test.shape[0]



model = build_model(train_type=1)

model.load_weights("../input/rsnaefnetb3v2/fold-0.h5")



for x in custom_dcom_image_generator(batch_size, test, True, False):

    clear_output(wait=True)

    

    preds = model.predict(x, batch_size=32, verbose=1)

    #print(preds)

    try:

        for key in preds.keys():

            predictions[key] += preds[key].flatten().tolist()

            

    except Exception as e:

        print(e)

        for key in preds.keys():

            predictions[key] = preds[key].flatten().tolist()

            

    l = (l+batch_size)%n

    print('Total predicted:', len(predictions['indeterminate']),'/', n)

    p, c = c, time.time()

    print("One batch time: %.2f seconds" %(c-p))

    print("ETA: %.2f" %((n-l)*(c-p)/batch_size))

    

    if c - pred_start_time >= stopper:

        print("Time's up!")

        break

    

    #del model

    #K.clear_session()

    

    del x, preds

    gc.collect()

    #break
for key in predictions.keys():

    print(key, np.array(predictions[key]).shape)
test_ids = []

for v in test.StudyInstanceUID:

    if v not in test_ids:

        test_ids.append(v)

        

test_preds = test.copy()

test_preds = pd.concat([test_preds, pd.DataFrame(predictions)], axis=1)

test_preds
IDS = []

labels = []



for label in ['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',

                 'leftsided_pe', 'chronic_pe', 'rightsided_pe',

                 'acute_and_chronic_pe', 'central_pe', 'indeterminate']:

    for key in test_ids:

        temp = test_preds.loc[test_preds.StudyInstanceUID==key]

        

        IDS.append('_'.join([key, label]))

        labels.append(np.max(temp[label]))
IDS += test_preds.SOPInstanceUID.tolist()

labels += test_preds['negative_exam_for_pe'].tolist()



sub = pd.DataFrame({"id":IDS, 'label':labels})

sub
sub.fillna(0.28, inplace=True)

sub.to_csv('submission.csv', index=False)
sub