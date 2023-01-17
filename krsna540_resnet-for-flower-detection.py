# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from skimage import io

import os

import glob

import numpy as np

import matplotlib.pyplot as plt

import random

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# function to plot n images using subplots

def plot_image(images, captions=None, cmap=None ):

    f, axes = plt.subplots(1, len(images), sharey=True)

    f.set_figwidth(15)

    for ax,image in zip(axes, images):

        ax.imshow(image, cmap)
# path to your dataset

DATASET_PATH = '/kaggle/input/flowers-recognition/flowers/'

flowers_cls = ['daisy', 'rose']
# globbing example

# help(glob)

flower_path = os.path.join(DATASET_PATH, flowers_cls[1], '*')

print(flower_path)



# glob through the directory (returns a list of all file paths)

flower_path = glob.glob(flower_path)

print(flower_path[3]) # access an individual file
rand_index = random.randint(0, len(flower_path))

image = io.imread(flower_path[rand_index])

plt.imshow(image)
# plot a sample image

flower_path = os.path.join(DATASET_PATH, flowers_cls[1], '*')

flower_path = glob.glob(flower_path)



# access some element (a file) from the list

image = io.imread(flower_path[729])

plt.imshow(image)

print(image.shape)
# plotting the original image and the RGB channels

f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)

f.set_figwidth(15)

ax1.imshow(image)



# RGB channels

ax2.imshow(image[:, : , 0])

ax3.imshow(image[:, : , 1])

ax4.imshow(image[:, : , 2])

f.suptitle('Different Channels of Image')
# bin_image will be a (240, 320) True/False array

bin_image = image[:, :, 0] > 125

plot_image([image, bin_image], cmap='gray')
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening

from skimage.morphology import selem



# use a disk of radius 3

selem = selem.disk(3)



# oprning and closing

open_img = binary_opening(bin_image, selem)

close_img = binary_closing(bin_image, selem)



# erosion and dilation

eroded_img = binary_erosion(bin_image, selem)

dilated_img = binary_dilation(bin_image, selem)



plot_image([bin_image, open_img, close_img, eroded_img, dilated_img], cmap='gray')
norm1_image = image/255

norm2_image = image - np.min(image)/np.max(image) - np.min(image)

norm3_image = image - np.percentile(image,5)/ np.percentile(image,95) - np.percentile(image,5)



plot_image([image, norm1_image, norm2_image, norm3_image], cmap='gray')
from skimage import transform as tf



# flip left-right, up-down

image_flipr = np.fliplr(image)

image_flipud = np.flipud(image)



plot_image([image, image_flipr, image_flipud])
# specify x and y coordinates to be used for shifting (mid points)

shift_x, shift_y = image.shape[0]/2, image.shape[1]/2



# translation by certain units

matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])

matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])



# rotation

rot_transforms =  tf.AffineTransform(rotation=np.deg2rad(45))

rot_matrix = matrix_to_topleft + rot_transforms + matrix_to_center

rot_image = tf.warp(image, rot_matrix)



# scaling 

scale_transforms = tf.AffineTransform(scale=(2, 2))

scale_matrix = matrix_to_topleft + scale_transforms + matrix_to_center

scale_image_zoom_out = tf.warp(image, scale_matrix)



scale_transforms = tf.AffineTransform(scale=(0.5, 0.5))

scale_matrix = matrix_to_topleft + scale_transforms + matrix_to_center

scale_image_zoom_in = tf.warp(image, scale_matrix)



# translation

transaltion_transforms = tf.AffineTransform(translation=(50, 50))

translated_image = tf.warp(image, transaltion_transforms)





plot_image([image, rot_image, scale_image_zoom_out, scale_image_zoom_in, translated_image])
# shear transforms

shear_transforms = tf.AffineTransform(shear=np.deg2rad(45))

shear_matrix = matrix_to_topleft + shear_transforms + matrix_to_center

shear_image = tf.warp(image, shear_matrix)



bright_jitter = image*0.999 + np.zeros_like(image)*0.001



plot_image([image, shear_image, bright_jitter])
import six

from keras.models import Model

from keras.layers import (

    Input,

    Activation,

    Dense,

    Flatten

)

from keras.layers.convolutional import (

    Conv2D,

    MaxPooling2D,

    AveragePooling2D

)

from keras.layers.merge import add

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras import backend as K





def _bn_relu(input):

    """Helper to build a BN -> relu block

    """

    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)

    return Activation("relu")(norm)





def _conv_bn_relu(**conv_params):

    """Helper to build a conv -> BN -> relu block

    """

    filters = conv_params["filters"]

    kernel_size = conv_params["kernel_size"]

    strides = conv_params.setdefault("strides", (1, 1))

    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

    padding = conv_params.setdefault("padding", "same")

    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))



    def f(input):

        conv = Conv2D(filters=filters, kernel_size=kernel_size,

                      strides=strides, padding=padding,

                      kernel_initializer=kernel_initializer,

                      kernel_regularizer=kernel_regularizer)(input)

        return _bn_relu(conv)



    return f





def _bn_relu_conv(**conv_params):

    """Helper to build a BN -> relu -> conv block.

    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf

    """

    filters = conv_params["filters"]

    kernel_size = conv_params["kernel_size"]

    strides = conv_params.setdefault("strides", (1, 1))

    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")

    padding = conv_params.setdefault("padding", "same")

    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))



    def f(input):

        activation = _bn_relu(input)

        return Conv2D(filters=filters, kernel_size=kernel_size,

                      strides=strides, padding=padding,

                      kernel_initializer=kernel_initializer,

                      kernel_regularizer=kernel_regularizer)(activation)



    return f





def _shortcut(input, residual):

    """Adds a shortcut between input and residual block and merges them with "sum"

    """

    # Expand channels of shortcut to match residual.

    # Stride appropriately to match residual (width, height)

    # Should be int if network architecture is correctly configured.

    input_shape = K.int_shape(input)

    residual_shape = K.int_shape(residual)

    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))

    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))

    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]



    shortcut = input

    # 1 X 1 conv if shape is different. Else identity.

    if stride_width > 1 or stride_height > 1 or not equal_channels:

        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],

                          kernel_size=(1, 1),

                          strides=(stride_width, stride_height),

                          padding="valid",

                          kernel_initializer="he_normal",

                          kernel_regularizer=l2(0.0001))(input)



    return add([shortcut, residual])





def _residual_block(block_function, filters, repetitions, is_first_layer=False):

    """Builds a residual block with repeating bottleneck blocks.

    """

    def f(input):

        for i in range(repetitions):

            init_strides = (1, 1)

            if i == 0 and not is_first_layer:

                init_strides = (2, 2)

            input = block_function(filters=filters, init_strides=init_strides,

                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)

        return input



    return f





def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):

    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.

    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    """

    def f(input):



        if is_first_block_of_first_layer:

            # don't repeat bn->relu since we just did bn->relu->maxpool

            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),

                           strides=init_strides,

                           padding="same",

                           kernel_initializer="he_normal",

                           kernel_regularizer=l2(1e-4))(input)

        else:

            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),

                                  strides=init_strides)(input)



        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        return _shortcut(input, residual)



    return f





def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):

    """Bottleneck architecture for > 34 layer resnet.

    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:

        A final conv layer of filters * 4

    """

    def f(input):



        if is_first_block_of_first_layer:

            # don't repeat bn->relu since we just did bn->relu->maxpool

            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),

                              strides=init_strides,

                              padding="same",

                              kernel_initializer="he_normal",

                              kernel_regularizer=l2(1e-4))(input)

        else:

            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),

                                     strides=init_strides)(input)



        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)

        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)

        return _shortcut(input, residual)



    return f





def _handle_dim_ordering():

    global ROW_AXIS

    global COL_AXIS

    global CHANNEL_AXIS

    if K.image_data_format() == 'tf':

        ROW_AXIS = 1

        COL_AXIS = 2

        CHANNEL_AXIS = 3

    else:

        CHANNEL_AXIS = 1

        ROW_AXIS = 2

        COL_AXIS = 3





def _get_block(identifier):

    if isinstance(identifier, six.string_types):

        res = globals().get(identifier)

        if not res:

            raise ValueError('Invalid {}'.format(identifier))

        return res

    return identifier





class ResnetBuilder(object):

    @staticmethod

    def build(input_shape, num_outputs, block_fn, repetitions):

        """Builds a custom ResNet like architecture.

        Args:

            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)

            num_outputs: The number of outputs at final softmax layer

            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.

                The original paper used basic_block for layers < 50

            repetitions: Number of repetitions of various block units.

                At each block unit, the number of filters are doubled and the input size is halved

        Returns:

            The keras `Model`.

        """

        _handle_dim_ordering()

        if len(input_shape) != 3:

            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")



        # Permute dimension order if necessary

        if K.image_data_format() == 'tf':

            input_shape = (input_shape[1], input_shape[2], input_shape[0])



        # Load function from str if needed.

        block_fn = _get_block(block_fn)



        input = Input(shape=input_shape)

        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)



        block = pool1

        filters = 64

        for i, r in enumerate(repetitions):

            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)

            filters *= 2



        # Last activation

        block = _bn_relu(block)



        # Classifier block

        block_shape = K.int_shape(block)

        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),

                                 strides=(1, 1))(block)

        flatten1 = Flatten()(pool2)

        dense = Dense(units=num_outputs, kernel_initializer="he_normal",

                      activation="softmax")(flatten1)



        model = Model(inputs=input, outputs=dense)

        return model



    @staticmethod

    def build_resnet_18(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])



    @staticmethod

    def build_resnet_34(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])



    @staticmethod

    def build_resnet_50(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])



    @staticmethod

    def build_resnet_101(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])



    @staticmethod

    def build_resnet_152(input_shape, num_outputs):

        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])
# specify image size and channels

img_channels = 3

img_rows = 100

img_cols = 100



# number of classes

nb_classes = 2
import numpy as np

import keras



class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    

    def __init__(self, mode='train', ablation=None, flowers_cls=['daisy', 'rose'], 

                 batch_size=32, dim=(100, 100), n_channels=3, shuffle=True):

        """

        Initialise the data generator

        """

        self.dim = dim

        self.batch_size = batch_size

        self.labels = {}

        self.list_IDs = []

        

        # glob through directory of each class 

        for i, cls in enumerate(flowers_cls):

            paths = glob.glob(os.path.join(DATASET_PATH, cls, '*'))

            brk_point = int(len(paths)*0.8)

            if mode == 'train':

                paths = paths[:brk_point]

            else:

                paths = paths[brk_point:]

            if ablation is not None:

                paths = paths[:ablation]

            self.list_IDs += paths

            self.labels.update({p:i for p in paths})

            

        self.n_channels = n_channels

        self.n_classes = len(flowers_cls)

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        X, y = self.__data_generation(list_IDs_temp)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        

        delete_rows = []



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            img = io.imread(ID)

            img = img/255

            if img.shape[0] > 100 and img.shape[1] > 100:

                h, w, _ = img.shape

                img = img[int(h/2)-50:int(h/2)+50, int(w/2)-50:int(w/2)+50, : ]

            else:

                delete_rows.append(i)

                continue

            

            X[i,] = img

          

            # Store class

            y[i] = self.labels[ID]

        

        X = np.delete(X, delete_rows, axis=0)

        y = np.delete(y, delete_rows, axis=0)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
# using resnet 18

model = ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)

model.compile(loss='categorical_crossentropy', optimizer='SGD',

              metrics=['accuracy'])



# create data generator objects in train and val mode

# specify ablation=number of data points to train on

training_generator = DataGenerator('train', ablation=100)

validation_generator = DataGenerator('val', ablation=100)



# fit: this will fit the net on 'ablation' samples, only 1 epoch

model.fit_generator(generator=training_generator,

                    validation_data=validation_generator,

                    epochs=1,)
tf.test.is_gpu_available()