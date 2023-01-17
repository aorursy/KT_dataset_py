import os

import json

import gc

import cv2

import six

import keras

import tensorflow as tf

from keras import backend as K

from keras import layers

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.optimizers import Adam

from keras.callbacks import Callback, ModelCheckpoint

from keras.regularizers import l2

from keras.engine.topology import Input

from keras.engine.training import Model

from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose

from keras.layers.core import Activation, SpatialDropout2D

from keras.layers.merge import concatenate,add

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D

from keras import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.models import load_model

from keras.optimizers import Adam

from keras.utils.vis_utils import plot_model

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization

from keras.layers import Conv2D, Concatenate, MaxPooling2D

from keras.layers import UpSampling2D, Dropout, BatchNormalization

from tqdm import tqdm_notebook

from keras import initializers

from keras import regularizers

from keras import constraints

from keras.utils import conv_utils

from keras.utils.data_utils import get_file

from keras.engine.topology import get_source_inputs

from keras.engine import InputSpec

from keras import backend as K

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import random

print(os.listdir('../input/severstal-steel-defect-detection/'))



#train_df 所有训练集图片的csv

train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

val_sub_df = train_df.copy()#use when predict val

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

# print('#train_df 所有训练集图片的csv')

# print(train_df.shape)

# print(train_df.head())

print('#val_sub_df 所有csv')

print(val_sub_df.head())







#repeat different class images

class_1_repeat = 32        # repeat class 1 examples x times

class_2_repeat = 60

class_3_repeat = 1

class_4_repeat = 32









class_1_img_id = train_df[((~ train_df['EncodedPixels'].isna())&(train_df['ClassId'] == '1'))]#897

class_1_img_id_index = np.repeat(class_1_img_id.index, class_1_repeat)

class_2_img_id = train_df[((~ train_df['EncodedPixels'].isna())&(train_df['ClassId'] == '2'))]#247

class_2_img_id_index = np.repeat(class_2_img_id.index, class_2_repeat)

class_3_img_id = train_df[((~ train_df['EncodedPixels'].isna())&(train_df['ClassId'] == '3'))]#5150

class_3_img_id_index = np.repeat(class_3_img_id.index, 1)

class_4_img_id = train_df[((~ train_df['EncodedPixels'].isna())&(train_df['ClassId'] == '4'))]#801

class_4_img_id_index = np.repeat(class_4_img_id.index, class_4_repeat)





repeated_train_image_ids = np.concatenate([class_2_img_id_index,class_1_img_id_index,class_3_img_id_index,class_4_img_id_index])

# repeated_train_image_ids = non_missing_train_idx.index

random.shuffle(repeated_train_image_ids)

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)





def rle2mask(rle, input_shape):

    width, height = input_shape[:2]



    mask = np.zeros(width * height).astype(np.uint8)



    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start + lengths[index])] = 1

        current_position += lengths[index]



    return mask.reshape(height, width).T





def build_masks(rles, input_shape):

    depth = len(rles)

    # print("rles depth :"+str(depth))#4

    masks = np.zeros((*input_shape, depth))



    for i, rle in enumerate(rles):

        if type(rle) is str:

            masks[:, :, i] = rle2mask(rle, input_shape)



    return masks





def build_rles(masks):

    width, height, depth = masks.shape



    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]



    return rles





class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'



    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path='../input/severstal-steel-defect-detection/train_images',

                 batch_size=32, dim=(256, 1600), n_channels=3,  ####输入的通道数该为3

                 n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.random_state = random_state



        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]



        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]



        X = self.__generate_X(list_IDs_batch)



        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            return X, y



        elif self.mode == 'predict':

            return X



        else:

            raise AttributeError('The mode parameter should be set to "fit" or "predict".')



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.seed(self.random_state)

            np.random.shuffle(self.indexes)



    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        X = np.empty((self.batch_size, *self.dim, self.n_channels))



        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            img_path = self.base_path+'/'+im_name#f"{self.base_path}/{im_name}"

            img = self.__load_rgb(img_path)  # 该为彩色3通道



            # Store samples

            X[i,] = img



        return X



    def __generate_y(self, list_IDs_batch):

        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)



        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]



            rles = image_df['EncodedPixels'].values

            masks = build_masks(rles, input_shape=self.dim)



            y[i,] = masks



        return y



    def __load_grayscale(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.

        img = np.expand_dims(img, axis=-1)



        return img



    def __load_rgb(self, img_path):

        #         print('load_rgb')

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.



        return img



BATCH_SIZE = 16



train_idx, val_idx = train_test_split(

    repeated_train_image_ids,#non_missing_train_idx.index,  # NOTICE DIFFERENCE

    random_state=2019,

    test_size=0.05

)



train_generator = DataGenerator(

    train_idx,

    df=train_df,

    target_df=train_df,

    batch_size=BATCH_SIZE,

    n_classes=4

)



val_generator = DataGenerator(

    val_idx,

    df=train_df,

    target_df=train_df,

    batch_size=BATCH_SIZE,

    n_classes=4

)





########build unet model ###########



def handle_block_names(stage):

    conv_name = 'decoder_stage{}_conv'.format(stage)

    bn_name = 'decoder_stage{}_bn'.format(stage)

    relu_name = 'decoder_stage{}_relu'.format(stage)

    up_name = 'decoder_stage{}_upsample'.format(stage)

    return conv_name, bn_name, relu_name, up_name





def Upsample2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),

                     batchnorm=False, skip=None):



    def layer(input_tensor):



        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)



        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)



        if skip is not None:

            x = Concatenate()([x, skip])



        x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'1')(x)

        if batchnorm:

            x = BatchNormalization(name=bn_name+'1')(x)

        x = Activation('relu', name=relu_name+'1')(x)



        x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'2')(x)

        if batchnorm:

            x = BatchNormalization(name=bn_name+'2')(x)

        x = Activation('relu', name=relu_name+'2')(x)



        return x

    return layer





def Transpose2D_block(filters, stage, kernel_size=(3,3), upsample_rate=(2,2),

                      transpose_kernel_size=(4,4), batchnorm=False, skip=None):



    def layer(input_tensor):



        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)



        x = Conv2DTranspose(filters, transpose_kernel_size, strides=upsample_rate,

                            padding='same', name=up_name)(input_tensor)

        if batchnorm:

            x = BatchNormalization(name=bn_name+'1')(x)

        x = Activation('relu', name=relu_name+'1')(x)



        if skip is not None:

            x = Concatenate()([x, skip])



        x = Conv2D(filters, kernel_size, padding='same', name=conv_name+'2')(x)

        if batchnorm:

            x = BatchNormalization(name=bn_name+'2')(x)

        x = Activation('relu', name=relu_name+'2')(x)



        return x

    return layer





def build_unet(backbone, classes, last_block_filters, skip_layers,

               n_upsample_blocks=5, upsample_rates=(2,2,2,2,2),

               block_type='upsampling', activation='sigmoid',

               **kwargs):



    input = backbone.input

    x = backbone.output



    if block_type == 'transpose':

        up_block = Transpose2D_block

    else:

        up_block = Upsample2D_block



    # convert layer names to indices

    skip_layers = ([get_layer_number(backbone, l) if isinstance(l, str) else l

                    for l in skip_layers])

    for i in range(n_upsample_blocks):



        # check if there is a skip connection

        if i < len(skip_layers):

#             print(backbone.layers[skip_layers[i]])

#             print(backbone.layers[skip_layers[i]].output)

            skip = backbone.layers[skip_layers[i]].output

        else:

            skip = None



        up_size = (upsample_rates[i], upsample_rates[i])

        filters = last_block_filters * 2**(n_upsample_blocks-(i+1))



        x = up_block(filters, i, upsample_rate=up_size, skip=skip, **kwargs)(x)



    if classes < 2:

        activation = 'sigmoid'

######################################################################################################################

    x = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)

    print('Conv2D classes',classes)

    x = Activation(activation, name=activation)(x)



    model = Model(input, x)



    return model



########build  resnet34 #######

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





def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):

    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.

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





def _handle_dim_ordering():

    global ROW_AXIS

    global COL_AXIS

    global CHANNEL_AXIS

    if K.image_dim_ordering() == 'tf':

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

    def build(input_shape, block_fn, repetitions, input_tensor):

        _handle_dim_ordering()

        if len(input_shape) != 3:

            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")



        # Permute dimension order if necessary

        if K.image_dim_ordering() == 'tf':

            input_shape = (input_shape[1], input_shape[2], input_shape[0])



        # Load function from str if needed.

        block_fn = _get_block(block_fn)



        if input_tensor is None:

            img_input = Input(shape=input_shape)

        else:

            if not K.is_keras_tensor(input_tensor):

                img_input = Input(tensor=input_tensor, shape=input_shape)

            else:

                img_input = input_tensor



        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(img_input)

        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)



        block = pool1

        filters = 64

        for i, r in enumerate(repetitions):

            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)

            filters *= 2



        # Last activation

        block = _bn_relu(block)



        model = Model(inputs=img_input, outputs=block)

        return model



    @staticmethod

    def build_resnet_34(input_shape, input_tensor):

        return ResnetBuilder.build(input_shape, basic_block, [3, 4, 6, 3], input_tensor)





##########unet  with resnet34 encoder #############

def UResNet34(input_shape=(None, None, 3), classes=4, decoder_filters=16, decoder_block_type='upsampling',

                       encoder_weights=None, input_tensor=None, activation='sigmoid', **kwargs):



    backbone = ResnetBuilder.build_resnet_34(input_shape=input_shape,input_tensor=input_tensor)



    skip_connections = list([97,54,25])  # for resnet 34

    model = build_unet(backbone, classes, decoder_filters,

                       skip_connections, block_type=decoder_block_type,

                       activation=activation, **kwargs)

    model.name = 'u-resnet34'



    return model

model = UResNet34(input_shape=(3,256,1600))

model.summary()





##########define loss function ################

from keras.losses import binary_crossentropy

from keras import backend as K





def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score

def focal_loss(y_true, y_pred):

    gamma = 0.75

    alpha = 0.25

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))



    pt_1 = K.clip(pt_1, 1e-3, .999)

    pt_0 = K.clip(pt_0, 1e-3, .999)



    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(

        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))





# Focal Loss + DICE LOSS

def mixedLoss(y_true, y_pred, alpha=0.01):

    return alpha * focal_loss(y_true, y_pred) - K.log(dice_loss(y_true, y_pred))



# #lovasz loss

# def lovasz_grad(gt_sorted):

#     """

#     Computes gradient of the Lovasz extension w.r.t sorted errors

#     See Alg. 1 in paper

#     """

#     gts = tf.reduce_sum(gt_sorted)

#     intersection = gts - tf.cumsum(gt_sorted)

#     union = gts + tf.cumsum(1. - gt_sorted)

#     jaccard = 1. - intersection / union

#     jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)

#     return jaccard

#

#

#

#

# def lovasz_hinge_flat(logits, labels):

#     """

#     Binary Lovasz hinge loss

#       logits: [P] Variable, logits at each prediction (between -\infty and +\infty)

#       labels: [P] Tensor, binary ground truth labels (0 or 1)

#       ignore: label to ignore

#     """

#

#     def compute_loss():

#         labelsf = tf.cast(labels, logits.dtype)

#         signs = 2. * labelsf - 1.

#         errors = 1. - logits * tf.stop_gradient(signs)

#         errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")

#         gt_sorted = tf.gather(labelsf, perm)

#         grad = lovasz_grad(gt_sorted)

#         loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")

#         return loss

#

#     # deal with the void prediction case (only void pixels)

#     loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),

#                    lambda: tf.reduce_sum(logits) * 0.,

#                    compute_loss,

#                    strict=True,

#                    name="loss"

#                    )

#     return loss





model.compile(loss=binary_crossentropy, optimizer="adadelta", metrics=["accuracy"])#dice_loss



#########training ##########################

# early_stopping = EarlyStopping(patience=10, verbose=1)

# model_checkpoint = ModelCheckpoint("./keras.model",

#                                    save_best_only=True,

#                                    verbose=0, save_weights_only=False,

#                                     mode='auto')

# reduce_lr = ReduceLROnPlateau(factor=0.1, patience=4, min_lr=0.00001, verbose=1)



# epochs = 20





# history = model.fit_generator( train_generator,

#     validation_data=val_generator,

#                     epochs=epochs,

#                     callbacks=[early_stopping, model_checkpoint, reduce_lr],

#                     use_multiprocessing=False,

#                     workers=1

#                    )#callbacks=[early_stopping, model_checkpoint, reduce_lr],





# print(history.history)

# with open('history.json', 'w+') as f:

#     historyDecoder = history.history

#     for k in historyDecoder.keys():

#         historyDecoder[k] = list(map(float, historyDecoder[k]))

#     json.dump((historyDecoder), f)

# print(historyDecoder)

# with open('history.json','r') as f:

#     history2 = json.load(f)

#     print(history2)

# history_df = pd.DataFrame(history2)

# history_df[['loss', 'val_loss']].plot()

# history_df[['acc', 'val_acc']].plot()

# # plt.show()





# predict val

# model = load_model('./keras.model')



val_image = train_df.loc[val_idx]

val_image = pd.DataFrame(val_image['ImageId'].unique(), columns=['ImageId'])

val_df = []



for i in range(0, val_image.shape[0], 500):

    batch_idx = list(

        range(i, min(val_image.shape[0], i + 500))

    )



    val_generator = DataGenerator(

        batch_idx,

        df=val_image,

        shuffle=False,

        mode='predict',

        base_path='../input/severstal-steel-defect-detection/train_images',

        target_df=val_sub_df,

        batch_size=1,

        n_classes=4

    )



    batch_pred_masks = model.predict_generator(

        val_generator,

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):

        filename = val_image['ImageId'].iloc[b]

        image_df = val_sub_df[val_sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j,].round().astype(int)

        pred_rles = build_rles(pred_masks)



        image_df['EncodedPixels'] = pred_rles

        val_df.append(image_df)



val_df = pd.concat(val_df)

val_df.drop(columns='ImageId', inplace=True)

val_df.to_csv('../input/severstal-steel-defect-detection/val_create.csv', index=False)



















# predict and submit

sub_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')

sub_df['ImageId'] = sub_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

# print('test_imgs  num'+test_imgs.shape)

test_df = []



for i in range(0, test_imgs.shape[0], 500):

    batch_idx = list(

        range(i, min(test_imgs.shape[0], i + 500))

    )



    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/severstal-steel-defect-detection/test_images',

        target_df=sub_df,

        batch_size=1,

        n_classes=4

    )



    batch_pred_masks = model.predict_generator(

        test_generator,

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):

        filename = test_imgs['ImageId'].iloc[b]

        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j,].round().astype(int)

        pred_rles = build_rles(pred_masks)



        image_df['EncodedPixels'] = pred_rles

        test_df.append(image_df)



test_df = pd.concat(test_df)

test_df.drop(columns='ImageId', inplace=True)

test_df.to_csv('../input/severstal-steel-defect-detection/submission.csv', index=False)






