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

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Model

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import plot_model

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications.resnet import preprocess_input

import numpy as np

import os

from PIL import Image

from glob import glob

import random

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

from skimage.transform import resize
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load Images from each category 

def loadBatchImages(path, nSamples):

    categoriesList = os.listdir(path)

    imagesTrain, labelsTrain = [], []

    for category in categoriesList[0:10]:

        deepPath = path + category + '/'

        imageList = os.listdir(deepPath)

        idx = 0

        for image in imageList[0:nSamples]:

            img = load_img(deepPath + image)

            img = img_to_array(img)

            img = resize(img, (224,224)) # - Use this if VGG is the model

            if idx < nSamples:

                labelsTrain.append(int(image[1:2]))

                imagesTrain.append(img)

            idx+=1

    return imagesTrain, to_categorical(labelsTrain)
train_path = '/kaggle/input/10-monkey-species/training/training/'

val_path = '/kaggle/input/10-monkey-species/validation/validation/'

imagesTrain, labelsTrain = loadBatchImages(train_path, 10)

imagesVal, labelsVal = loadBatchImages(val_path, 10)
def shuffleData(a, b):

    assert np.shape(a)[0] == np.shape(b)[0]

    p = np.random.permutation(np.shape(a)[0])

    return (a[p], b[p])
data = [preprocess_input(np.float64(image)) for image in imagesTrain]

dataVal = [preprocess_input(np.float64(image)) for image in imagesVal]

train = shuffleData(np.asarray(data),labelsTrain)

val = shuffleData(np.asarray(dataVal),labelsVal)

x_train = train[0]

y_train = train[1]

x_test = val[0]

y_test = val[1]
x_train.shape, y_train.shape, x_test.shape, y_test.shape
multipleImages = glob('/kaggle/input/10-monkey-species/training/training/n6/**')

def plotThreeImages(images):

    r = random.sample(images, 3)

    plt.figure(figsize=(16,16))

    plt.subplot(131)

    plt.imshow(cv2.imread(r[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(r[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(r[2])); 

plotThreeImages(multipleImages)

plotThreeImages(multipleImages)

plotThreeImages(multipleImages)
# training parameters

batch_size = 32 # orig paper trained all networks with batch_size=128

epochs = 200

data_augmentation = True

num_classes = 10



# subtracting pixel mean improves accuracy

subtract_mean_pixel = True

# Any results you write to the current directory are saved as output.
# Model parameter

# ----------------------------------------------------------------------------

#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch

# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti

#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)

# ----------------------------------------------------------------------------

# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)

# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)

# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)

# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)

# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)

# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)

# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)

# ---------------------------------------------------------------------------

n = 3
# model version

# orig paper: version = 1 (ResNet v1), 

# improved ResNet: version = 2 (ResNet v2)

version = 1
# computed depth from supplied model parameter n

if version == 1:

    depth = n * 6 + 2

elif version == 2:

    depth = n * 9 + 2

# model name, depth and version

model_type = 'ResNet%dv%d' % (depth, version)

model_type
# normalize data.

x_train = x_train / 255.
x_test = x_test / 255.
# if subtract pixel mean is enabled

if subtract_mean_pixel:

    x_train_mean = np.mean(x_train, axis=0)

    x_test_mean = np.mean(x_test, axis=0)

    x_train -= x_train_mean

    x_test -= x_test_mean
print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape)
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

model = resnet_v1(input_shape=(224,224,3),  depth=depth)

model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=lr_schedule(0)),

              metrics=['accuracy'])

model.summary()
plot_model(model, to_file="%s.png" % model_type, show_shapes=True)

print(model_type)
# prepare model model saving directory.

save_dir = os.path.join(os.getcwd(), 'saved_models')

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