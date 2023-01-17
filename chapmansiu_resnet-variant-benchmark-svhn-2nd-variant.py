import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_digits

import  tensorflow  as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model
from keras.layers import Conv1D, Conv2D, Conv3D, SeparableConv1D, MaxPooling1D, MaxPooling2D

from keras import backend as K
from keras import initializers, layers
from keras.engine.topology import Layer

from keras.utils import to_categorical

from sklearn import metrics
from sklearn import tree, ensemble
import scipy.io as sio
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
try:
    from keras_applications.resnet50 import identity_block, conv_block
except:
    from keras.applications.resnet50 import identity_block, conv_block
from keras import utils as keras_utils
train_dat = sio.loadmat("../input/train_32x32.mat")

n_samples = train_dat['X'].T.shape[0]
X = train_dat['X'].T
X = np.swapaxes(X, 1, 3)
y_ = train_dat['y'].flatten()
y = to_categorical(y_)

test_dat = sio.loadmat("../input/test_32x32.mat")
#n_samples = test_dat['X'].T.shape[0]
X_test = test_dat['X'].T
X_test = np.swapaxes(X_test, 1, 3)
y_test_ = test_dat['y'].flatten()
y_test = to_categorical(y_test_)

nepochs=5
nclasses = y.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def ResNet(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    """
    Simplified resnet for a 32x32x3 image. 
    
    For testing/benchmark purposes only
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=32,
    #                                   data_format=backend.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)

    #if input_tensor is None:
    #    img_input = layers.Input(shape=input_shape)
    #else:
    #    if not backend.is_keras_tensor(input_tensor):
    #       img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    #    else:
    #        img_input = input_tensor
    img_input = input_tensor
    bn_axis = 3
    #if backend.image_data_format() == 'channels_last':
    #    bn_axis = 3
    #else:
    #    bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    ## uncomment below for more complex images
    
    #x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    #x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    #x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    #x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    #x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    #x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    #if input_tensor is not None:
    #    inputs = keras_utils.get_source_inputs(input_tensor)
    #else:
    #    inputs = img_input
    inputs = input_tensor
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model
# using resnet50 as benchmark
#from keras.applications.resnet50 import ResNet50

main_input = Input(shape=(32, 32, 3), name='main_input')
# replace with resnet50 if you wish
model = ResNet(include_top=True, weights=None, input_tensor=main_input, input_shape=(32, 32, 3), pooling='max', classes=nclasses)

model.compile(optimizer='adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

hist = model.fit(X, y, validation_data=(X_test, y_test), epochs=5, verbose=1)
print(pd.DataFrame(hist.history).iloc[-1])


