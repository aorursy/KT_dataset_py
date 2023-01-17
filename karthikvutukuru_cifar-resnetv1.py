# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



from tensorflow.keras.layers import Dense, Conv2D

from tensorflow.keras.layers import BatchNormalization, Activation

from tensorflow.keras.layers import AveragePooling2D, Input

from tensorflow.keras.layers import Flatten, add

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Model

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import plot_model

from tensorflow.keras.utils import to_categorical

import numpy as np

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Training Parameters

batch_size = 32

epochs = 200

data_augmentation = True

num_classes = 10

# subtracting pixel mean improves accuracy

subtract_pixel_mean = True
n = 3

# model version

# orig paper: version = 1 (ResNet v1), 

# improved ResNet: version = 2 (ResNet v2)

version = 1
# computed depth from supplied parameter

if version == 1:

    depth = 6 * n + 2

else:

    depth = 9 *n + 2

    
# Model name, type and version

model_type = 'ResNet {}v{}'.format(depth, version)

model_type
# Load Data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input Image Dimensions

input_shape = x_train.shape[1:]

input_shape
# Normalize the inputs

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255
# If subtract_mean is enabled

if subtract_pixel_mean:

    x_train_mean = np.mean(x_train, axis=0)

    x_train -= x_train_mean

    x_test -= x_train_mean

    
print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape)
x_test.shape
# Convert y to one hot encoding

y_train = to_categorical(y_train, num_classes)

y_test = to_categorical(y_test, num_classes)

y_train.shape, y_test.shape
def lr_schedule(epoch):

    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.

    Called automatically every epoch as part of callbacks during training.

    # Arguments

        epoch (int): The number of epochs

    # Returns

        lr (float32): learning rate

    """

    

    lr = 1e-3

    if epoch > 180:

        lr *= 0.5e-3

    elif epoch > 160:

        lr *= 1e-3

    elif epoch > 120:

        lr *= 1e-2

    elif epoch > 80:

        lr *= 1e-1

    print('Learning rate: ', lr)

    return lr
def resnet_layer(inputs,

                num_filters=16,

                kernel_size=3,

                strides=1,

                activation='relu',

                batch_normalization=True,

                conv_first=True):

    """2D Convolution-Batch Normalization-Activation stack builder

    Arguments:

        inputs (tensor): input tensor from input image or previous layer

        num_filters (int): Conv2D number of filters

        kernel_size (int): Conv2D square kernel dimensions

        strides (int): Conv2D square stride dimensions

        activation (string): activation name

        batch_normalization (bool): whether to include batch normalization

        conv_first (bool): conv-bn-activation (True) or

            bn-activation-conv (False)

    Returns:

        x (tensor): tensor as input to the next layer

    """

    conv = Conv2D(num_filters,

                 kernel_size=kernel_size,

                 strides=strides,

                 padding='same',

                 kernel_initializer='he_normal',

                 kernel_regularizer=l2(1e-4))

    x = inputs

    if conv_first:

        x = conv(x)

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

    else:

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

        x = conv(x)

        

    return x
def resnet_v1(input_shape, depth, num_classes=10):

    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU

    Last ReLU is after the shortcut connection.

    At the beginning of each stage, the feature map size is halved

    (downsampled) by a convolutional layer with strides=2, while 

    the number of filters is doubled. Within each stage, 

    the layers have the same number filters and the

    same number of filters.

    Features maps sizes:

    stage 0: 32x32, 16

    stage 1: 16x16, 32

    stage 2:  8x8,  64

    The Number of parameters is approx the same as Table 6 of [a]:

    ResNet20 0.27M

    ResNet32 0.46M

    ResNet44 0.66M

    ResNet56 0.85M

    ResNet110 1.7M

    Arguments:

        input_shape (tensor): shape of input image tensor

        depth (int): number of core convolutional layers

        num_classes (int): number of classes (CIFAR10 has 10)

    Returns:

        model (Model): Keras model instance

    """

    if (depth - 2) % 6 != 0:

        raise ValueError('depth should be 6n+2 (eg 20, 32, in [a])')

    # start model definition.

    num_filters = 16

    num_res_blocks = int((depth - 2) / 6)



    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs)

    # instantiate the stack of residual units

    for stack in range(3):

        for res_block in range(num_res_blocks):

            strides = 1

            # first layer but not first stack

            if stack > 0 and res_block == 0:  

                strides = 2  # downsample

            y = resnet_layer(inputs=x,

                             num_filters=num_filters,

                             strides=strides)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters,

                             activation=None)

            # first layer but not first stack

            if stack > 0 and res_block == 0:

                # linear projection residual shortcut

                # connection to match changed dims

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 batch_normalization=False)

            x = add([x, y])

            x = Activation('relu')(x)

        num_filters *= 2



    # add classifier on top.

    # v1 does not use BN after last shortcut connection-ReLU

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,

                    activation='softmax',

                    kernel_initializer='he_normal')(y)



    # instantiate model.

    model = Model(inputs=inputs, outputs=outputs)

    return model
filters=16

# Build the model based on version

if version == 2:

    model = resnet_v2(input_shape=input_shape, depth=depth)

else:

    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=lr_schedule(0)),

              metrics=['accuracy'])

model.summary()
plot_model(model, to_file="/kaggle/working/%s.png" % model_type, show_shapes=True)

print(model_type)
# prepare model model saving directory.

save_dir = os.path.join('/kaggle/working', 'saved_models')

model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)
# prepare callbacks for model saving and for learning rate adjustment.

checkpoint = ModelCheckpoint(filepath=filepath,

                             monitor='val_accuracy',

                             verbose=1,

                             save_best_only=True)



lr_scheduler = LearningRateScheduler(lr_schedule)



lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),

                               cooldown=0,

                               patience=5,

                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]
# run training, with or without data augmentation.

if not data_augmentation:

    print('Not using data augmentation.')

    model.fit(x_train, y_train,

              batch_size=batch_size,

              epochs=epochs,

              validation_data=(x_test, y_test),

              shuffle=True,

              callbacks=callbacks)

else:

    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation:

    datagen = ImageDataGenerator(

        # set input mean to 0 over the dataset

        featurewise_center=False,

        # set each sample mean to 0

        samplewise_center=False,

        # divide inputs by std of dataset

        featurewise_std_normalization=False,

        # divide each input by its std

        samplewise_std_normalization=False,

        # apply ZCA whitening

        zca_whitening=False,

        # randomly rotate images in the range (deg 0 to 180)

        rotation_range=0,

        # randomly shift images horizontally

        width_shift_range=0.1,

        # randomly shift images vertically

        height_shift_range=0.1,

        # randomly flip images

        horizontal_flip=True,

        # randomly flip images

        vertical_flip=False)



    # compute quantities required for featurewise normalization

    # (std, mean, and principal components if ZCA whitening is applied).

    datagen.fit(x_train)



    # fit the model on the batches generated by datagen.flow().

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

                        validation_data=(x_test, y_test),

                        epochs=epochs, verbose=1, 

                        steps_per_epoch=len(x_train)//batch_size,

                        callbacks=callbacks)
# score trained model

scores = model.evaluate(x_test,

                        y_test,

                        batch_size=batch_size,

                        verbose=0)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])