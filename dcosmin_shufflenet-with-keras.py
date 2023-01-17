# Copyrights by Radu Dogaru & Ioana Dogaru 
# ShuffleNet with Keras 
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
import numpy as np # linear algebra
import keras.backend as K 
import time as ti 
import cv2
import os
import glob # for including images
import scipy.io as sio
from sklearn.metrics import classification_report, confusion_matrix
from keras import layers
from keras import models
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import * 
from keras.callbacks import *
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.activations import *
import keras.backend as K
X_train = [] # training fruit images
y_train = [] # training fruit labels 

X_test = [] # test fruit images
y_test = [] # test fruit labels 
# Training dataset
# We will need the images in a 32x32x3 input format.


for dir_path in glob.glob("../input/fruits/fruits-360/Training/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_train.append(img)
        y_train.append(img_label)
        
X_train=np.array(X_train)
y_train=np.array(y_train)
# Test dataset 
# Images will also be in a 32x32x3 format.

X_test = [] # test fruit images
y_test = [] # test fruit labels 

for dir_path in glob.glob("../input/fruits/fruits-360/Test/*"):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_test.append(img)
        y_test.append(img_label)

X_test=np.array(X_test)
y_test=np.array(y_test)
X_train = X_train/255.0
X_test = X_test/255.0
print(y_test)
# Now we need to have them labeled with numbers from 0 - 131 
label_to_id={v:k for k, v in enumerate(np.unique(y_train))}
#print(label_to_id)

y_train_label_id = np.array([label_to_id[i] for i in y_train])
y_test_label_id = np.array([label_to_id[i] for i in y_test])

# We need to translate this to be "one hot encoded" so our CNN can understand, 
# otherwise it will think this is some sort of regression problem on a continuous axis

from keras.utils.np_utils import to_categorical
print(y_train_label_id.shape)

y_cat_train_label_id=to_categorical(y_train_label_id)
y_cat_test_label_id=to_categorical(y_test_label_id)
#!pip install utils
# Source code: https://github.com/opconty/keras-shufflenetV2/blob/master/shufflenetv2.py



from keras.utils import plot_model
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Input, Conv2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dense
from keras.models import Model
import keras.backend as K



import os
from keras import backend as K
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D,Input, Dense
from keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
import numpy as np


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x


def ShuffleNetV2(include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(32,32,3),
                 load_model=None,
                 num_shuffle_units=[3,7,3],
                 bottleneck_ratio=1,
                 classes=131):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=28, require_flatten=include_top,
                                      data_format=K.image_data_format())
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)

    return model
model = ShuffleNetV2(include_top=True, input_shape=(32, 32, 3),load_model=None, classes=131)
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
# ShuffleNet function

# def _stage(tensor, nb_groups, in_channels, out_channels, repeat):
#     x = _shufflenet_unit(tensor, nb_groups, in_channels, out_channels, 2)

#     for _ in range(repeat):
#         x = _shufflenet_unit(x, nb_groups, out_channels, out_channels, 1)

#     return x


# def _pw_group(tensor, nb_groups, in_channels, out_channels):
#     """Pointwise grouped convolution."""
#     nb_chan_per_grp = in_channels // nb_groups

#     pw_convs = []
#     for grp in range(nb_groups):
#         x = Lambda(lambda x: x[:, :, :, nb_chan_per_grp * grp: nb_chan_per_grp * (grp + 1)])(tensor)
#         grp_out_chan = int(out_channels / nb_groups + 0.5)

#         pw_convs.append(
#             Conv2D(grp_out_chan,
#                    kernel_size=(1, 1),
#                    padding='same',
#                    use_bias=False,
#                    strides=1)(x)
#         )

#     return Concatenate(axis=-1)(pw_convs)


# def _shuffle(x, nb_groups):
#     def shuffle_layer(x):
#         _, w, h, n = K.int_shape(x)
#         nb_chan_per_grp = n // nb_groups

#         x = K.reshape(x, (-1, w, h, nb_chan_per_grp, nb_groups))
#         x = K.permute_dimensions(x, (0, 1, 2, 4, 3)) # Transpose only grps and chs
#         x = K.reshape(x, (-1, w, h, n))

#         return x

#     return Lambda(shuffle_layer)(x)


# def _shufflenet_unit(tensor, nb_groups, in_channels, out_channels, strides, shuffle=True, bottleneck=4):
#     bottleneck_channels = out_channels // bottleneck

#     x = _pw_group(tensor, nb_groups, in_channels, bottleneck_channels)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     if shuffle:
#         x = _shuffle(x, nb_groups)

#     x = DepthwiseConv2D(kernel_size=(3, 3),
#                         padding='same',
#                         use_bias=False,
#                         strides=strides)(x)
#     x = BatchNormalization()(x)


#     x = _pw_group(x, nb_groups, bottleneck_channels,
#                   out_channels if strides < 2 else out_channels - in_channels)
#     x = BatchNormalization()(x)

#     if strides < 2:
#         x = Add()([tensor, x])
#     else:
#         avg = AveragePooling2D(pool_size=(3, 3),
#                                strides=2,
#                                padding='same')(tensor)

#         x = Concatenate(axis=-1)([avg, x])

#     x = Activation('relu')(x)

#     return x


# def _info(nb_groups):
#     return {
#         1: [24, 144, 288, 576],
#         2: [24, 200, 400, 800],
#         3: [24, 240, 480, 960],
#         4: [24, 272, 544, 1088],
#         8: [24, 384, 768, 1536]
#     }[nb_groups], [None, 3, 7, 3]


# def ShuffleNet(input_shape, nb_classes, include_top=True, weights=None, nb_groups=8):
#     x_in = Input(shape=input_shape)

#     x = Conv2D(24,
#                kernel_size=(3, 3),
#                strides=2,
#                use_bias=False,
#                padding='same')(x_in)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     x = MaxPooling2D(pool_size=(3, 3),
#                      strides=2,
#                      padding='same')(x)

#     channels_list, repeat_list = _info(nb_groups)
#     for i, (out_channels, repeat) in enumerate(zip(channels_list[1:], repeat_list[1:]), start=1):
#         x = _stage(x, nb_groups, channels_list[i-1], out_channels, repeat)

#     if include_top:
#         x = GlobalAveragePooling2D()(x)
#         x = Dense(nb_classes, activation='softmax')(x)

#     model = Model(inputs=x_in, outputs=x)

#     if weights is not None:
#         model.load_weights(weights, by_name=True)

#     return model
# num_classes = 131
# print(np.shape(X_train)[1:4])
# print(np.shape(X_train))
# model=ShuffleNet(np.shape(X_train)[1:4], num_classes, include_top=True, weights=None)
# print('ShuffleNet implemented')
# model.compile(loss='categorical_crossentropy', 
#               optimizer='adam',
#               metrics=['accuracy'])
# print(model.summary())
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

result = model.fit(X_train,y_cat_train_label_id,
                       batch_size=15,
                       epochs=30,
                       verbose=1,
                       validation_data=(X_test,y_cat_test_label_id),
                       callbacks=callbacks_list
                      )
plt.figure(1)  
plt.plot(result.history['accuracy'])  
plt.plot(result.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()
plt.plot(result.history['loss'])  
plt.plot(result.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
plt.show()
# Load weights
model.load_weights("weights-improvement-28-0.97.hdf5")

# Compile model (required to make predictions)
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
print("Created model and loaded weights from file")
model.evaluate(X_test,y_cat_test_label_id)
#model.save("ShuffleNetV2 - 131.h5")