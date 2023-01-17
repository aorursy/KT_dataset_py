import pandas as pd, numpy as np, gc

from kaggle_datasets import KaggleDatasets

import tensorflow as tf, re, math

import tensorflow.keras.backend as K

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

import cv2

import audioread

import logging

import os

import random

import time

import warnings



import librosa

import numpy as np

import pandas as pd

import soundfile as sf

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.utils.data as data



from contextlib import contextmanager

from pathlib import Path

from typing import Optional



from fastprogress import progress_bar

from sklearn.metrics import f1_score

from torchvision import models
import os

import time



import numpy as np

from multiprocessing import Pool

import librosa

import scipy.signal

import sklearn

import cv2

import matplotlib.pyplot as plt
def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)  # type: ignore

    torch.backends.cudnn.deterministic = True  # type: ignore

    torch.backends.cudnn.benchmark = True  # type: ignore

    

    

def get_logger(out_file=None):

    logger = logging.getLogger()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger.handlers = []

    logger.setLevel(logging.INFO)



    handler = logging.StreamHandler()

    handler.setFormatter(formatter)

    handler.setLevel(logging.INFO)

    logger.addHandler(handler)



    if out_file is not None:

        fh = logging.FileHandler(out_file)

        fh.setFormatter(formatter)

        fh.setLevel(logging.INFO)

        logger.addHandler(fh)

    logger.info("logger set up")

    return logger

    

    

@contextmanager

def timer(name: str, logger: Optional[logging.Logger] = None):

    t0 = time.time()

    msg = f"[{name}] start"

    if logger is None:

        print(msg)

    else:

        logger.info(msg)

    yield



    msg = f"[{name}] done in {time.time() - t0:.2f} s"

    if logger is None:

        print(msg)

    else:

        logger.info(msg)
model_config = {

    "base_model_name": "resnet50",

    "pretrained": False,

    "num_classes": 264

}



melspectrogram_parameters = {

    "n_mels": 128,

    "fmin": 20,

    "fmax": 16000

}



weights_path = "../input/birdcall-resnet50-init-weights/best.pth"
logger = get_logger("main.log")

set_seed(1213)
TARGET_SR = 32000

TEST = Path("../input/birdsong-recognition/test_audio").exists()


if TEST:

    DATA_DIR = Path("../input/birdsong-recognition/")

else:

    DATA_DIR = Path("../input/birdcall-check/")

    

test = pd.read_csv(DATA_DIR / "test.csv")

test_audio = DATA_DIR / "test_audio"



test.head()
sub = pd.read_csv("../input/birdsong-recognition/sample_submission.csv")

sub.to_csv("submission.csv", index=False)  # this will be overwritten if everything goes well
BIRD_CODE = {

    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,

    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,

    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,

    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,

    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,

    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,

    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,

    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,

    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,

    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,

    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,

    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,

    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,

    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,

    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,

    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,

    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,

    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,

    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,

    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,

    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,

    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,

    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,

    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,

    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,

    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,

    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,

    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,

    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,

    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,

    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,

    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,

    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,

    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,

    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,

    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,

    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,

    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,

    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,

    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,

    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,

    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,

    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,

    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,

    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,

    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,

    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,

    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,

    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,

    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,

    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,

    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,

    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263

}



INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}
def binary_loss(y_true, y_pred):

    bce = K.binary_crossentropy(y_true, y_pred)

    return K.sum(bce, axis=-1)
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

  x = layers.Rescaling(1. / 1.)(x)

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

      name='stem_conv')(x)

  x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)

  x = layers.Activation(activation, name='stem_activation')(x)



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

      name='top_conv')(x)

  x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)

  x = layers.Activation(activation, name='top_activation')(x)

  if include_top:

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    if dropout_rate > 0:

      x = layers.Dropout(dropout_rate, name='top_dropout')(x)

    imagenet_utils.validate_activation(classifier_activation, weights)

    x = layers.Dense(

        classes,

        activation=classifier_activation,

        kernel_initializer=DENSE_KERNEL_INITIALIZER,

        name='predictions')(x)

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

        name=name + 'expand_conv')(

            inputs)

    x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)

    x = layers.Activation(activation, name=name + 'expand_activation')(x)

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

        name=name + 'se_reduce')(

            se)

    se = layers.Conv2D(

        filters,

        1,

        padding='same',

        activation='sigmoid',

        kernel_initializer=CONV_KERNEL_INITIALIZER,

        name=name + 'se_expand')(se)

    x = layers.multiply([x, se], name=name + 'se_excite')



  # Output phase

  x = layers.Conv2D(

      filters_out,

      1,

      padding='same',

      use_bias=False,

      kernel_initializer=CONV_KERNEL_INITIALIZER,

      name=name + 'project_conv')(x)

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

      0.2,

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
def build_model():

    inp = tf.keras.layers.Input(shape=(128,313,3))

    #base = tf.keras.applications.Xception(include_top=False, weights=None)

    base = EfficientNetB3(include_top=False,

                   weights=None,

                   input_shape=[128,313,3]

                         )

    x = base(inp)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(264,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inp,outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt,loss=binary_loss,metrics=[])

    return model
model = build_model()
model.load_weights("../input/birdcallefficientnetb3/fold-0f.h5")
def rescale(x):

    if np.max(x)-np.min(x)>1e-8:

        return (x-np.min(x))/(np.max(x)-np.min(x))#*255.

    else:

        return (x-np.min(x))/(1e-8)#*255.

    

def preEmphasis(signal, p=0.97):

    return scipy.signal.lfilter([1.0, -p], 1, signal)



def mono_to_color(X: np.ndarray,

                  Y: np.ndarray,

                  Z: np.ndarray,

                  mean=None,

                  std=None,

                  norm_max=None,

                  norm_min=None,

                  eps=1e-6):



    X = np.stack([Z, Y, X], axis=-1)

    

    for j in range(3):

        X[:,:,j] = rescale(X[:,:,j])

        



    return X*255

   





class TestDataset(data.Dataset):

    def __init__(self, df: pd.DataFrame, clip: np.ndarray,

                 img_size=313, melspectrogram_parameters={}):

        self.df = df

        self.clip = clip

        self.img_size = img_size

        self.melspectrogram_parameters = melspectrogram_parameters

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx: int):

        SR = 32000

        sample = self.df.loc[idx, :]

        site = sample.site

        row_id = sample.row_id

        

        if site == "site_3":

            y = self.clip.astype(np.float32)

            len_y = len(y)

            start = 0

            end = SR * 5

            images = []

            while len_y > start:

                y_batch = y[start:end].astype(np.float32)

                if len(y_batch) != (SR * 5):

                    break

                start = end

                end = end + SR * 5

                y_batch = preEmphasis(y_batch)

                y_batch = y_batch / np.max(np.abs(y_batch))

                melspec = librosa.feature.melspectrogram(y_batch,

                                                         sr=SR,

                                                         **self.melspectrogram_parameters)

                melspec = librosa.power_to_db(melspec).astype(np.float32)

                image = mono_to_color(melspec,melspec,melspec)

                height, width, _ = image.shape

                image = (image / 255.0).astype(np.float32)

                images.append(image)

            images = np.asarray(images)

            return images, row_id, site

        else:

            end_seconds = int(sample.seconds)

            start_seconds = int(end_seconds - 5)

            

            start_index = SR * start_seconds

            end_index = SR * end_seconds

            

            y = self.clip[start_index:end_index].astype(np.float32)

            y = preEmphasis(y)

            y = y / np.max(np.abs(y))

            melspec = librosa.feature.melspectrogram(y, sr=SR, **self.melspectrogram_parameters)

            melspec = librosa.power_to_db(melspec).astype(np.float32)

            image = mono_to_color(melspec,melspec,melspec)

            height, width, _ = image.shape

            image = (image / 255.0).astype(np.float32)



            return image, row_id, site
device = torch.device("cuda")

def prediction_for_clip(test_df: pd.DataFrame, 

                        clip: np.ndarray, 

                        

                        mel_params: dict, 

                        threshold=0.5):



    dataset = TestDataset(df=test_df, 

                          clip=clip,

                          img_size=313,

                          melspectrogram_parameters=mel_params)

    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    

    

    #model.eval()

    prediction_dict = {}

    for image, row_id, site in progress_bar(loader):

        site = site[0]

        row_id = row_id[0]

        if site in {"site_1", "site_2"}:

            image = image.to(device)



            image = image.to('cpu').detach().numpy().copy()

            proba = model.predict(image).reshape(-1)

            #print(proba[0])



            events = proba >= threshold

            labels = np.argwhere(events).reshape(-1).tolist()



        else:

            # to avoid prediction on large batch

            image = image.squeeze(0)

            batch_size = 16

            whole_size = image.size(0)

            if whole_size % batch_size == 0:

                n_iter = whole_size // batch_size

            else:

                n_iter = whole_size // batch_size + 1

                

            all_events = set()

            for batch_i in range(n_iter):

                batch = image[batch_i * batch_size:(batch_i + 1) * batch_size]

                if batch.ndim == 3:

                    batch = batch.unsqueeze(0)



                batch = batch.to(device)

                batch = batch.to('cpu').detach().numpy().copy()

                proba = model.predict(batch)

                    

                events = proba >= threshold

                for i in range(len(events)):

                    event = events[i, :]

                    labels = np.argwhere(event).reshape(-1).tolist()

                    for label in labels:

                        all_events.add(label)

                        

            labels = list(all_events)

        if len(labels) == 0:

            prediction_dict[row_id] = "nocall"

        else:

            labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))

            label_string = " ".join(labels_str_list)

            prediction_dict[row_id] = label_string

    return prediction_dict
def prediction(test_df: pd.DataFrame,

               test_audio: Path,

               model_config: dict,

               mel_params: dict,

               weights_path: str,

               threshold=0.5):

    #model = get_model(model_config, weights_path)

    unique_audio_id = test_df.audio_id.unique()



    warnings.filterwarnings("ignore")

    prediction_dfs = []

    for audio_id in unique_audio_id:

        with timer(f"Loading {audio_id}", logger):

            clip, _ = librosa.load(test_audio / (audio_id + ".mp3"),

                                   sr=TARGET_SR,

                                   mono=True,

                                   res_type="kaiser_fast")

        

        test_df_for_audio_id = test_df.query(

            f"audio_id == '{audio_id}'").reset_index(drop=True)

        with timer(f"Prediction on {audio_id}", logger):

            prediction_dict = prediction_for_clip(test_df_for_audio_id,

                                                  clip=clip,

                                                  

                                                  mel_params=mel_params,

                                                  threshold=threshold)

        row_id = list(prediction_dict.keys())

        birds = list(prediction_dict.values())

        prediction_df = pd.DataFrame({

            "row_id": row_id,

            "birds": birds

        })

        prediction_dfs.append(prediction_df)

    

    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)

    return prediction_df
submission = prediction(test_df=test,

                        test_audio=test_audio,

                        model_config=model_config,

                        mel_params=melspectrogram_parameters,

                        weights_path=weights_path,

                        threshold=0.5)

submission.to_csv("submission.csv", index=False)
submission