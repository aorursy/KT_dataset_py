BATCH_SIZE = 48

EDGE_CROP = 16

GAUSSIAN_NOISE = 0.1

UPSAMPLE_MODE = 'SIMPLE'

# downsampling inside the network

NET_SCALING = (1, 1)

# downsampling in preprocessing

IMG_SCALING = (2, 2)

# number of validation images to use

VALID_IMG_COUNT = 900

# maximum number of steps_per_epoch in training

MAX_TRAIN_STEPS = 9

MAX_TRAIN_EPOCHS = 99

AUGMENT_BRIGHTNESS = False
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread

import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap

from skimage.segmentation import mark_boundaries

# from skimage.util import montage2d as montage

from skimage.util import montage

from skimage.morphology import binary_opening, disk, label

import gc; gc.enable() 
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

ship_dir = '../input/mydata/data/'

train_image_dir = os.path.join(ship_dir, 'thermal_8_bt2')

mask_image_dir = os.path.join(ship_dir, 'mask')
masks = pd.read_csv(os.path.join('../input/mydata/', 'train_ship_segmentations_v3.csv'))

not_empty = pd.notna(masks.EncodedPixels)

print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')#非空图片中的mask数量

print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')#所有图片中非空图片

masks.head()
from sklearn.model_selection import train_test_split

train_ids, valid_ids = train_test_split(masks, 

                 test_size = 0.2, 

                 )

#stratify使训练和测试的ships比例一样

train_df = pd.merge(masks, train_ids)

valid_df = pd.merge(masks, valid_ids)

print(train_df.shape[0], 'training masks')

print(valid_df.shape[0], 'validation masks')
from PIL import Image
def make_image_gen(in_df, batch_size = BATCH_SIZE):

    all_batches = list(in_df.groupby('ImageId'))

    out_rgb = []

    out_mask = []

    while True:

        np.random.shuffle(all_batches)

        for c_img_id, c_masks in all_batches:

            rgb_path = os.path.join(train_image_dir, c_img_id)

            biaoqian_path = os.path.join(mask_image_dir, c_img_id)

            

            c_img1=Image.open(rgb_path).convert('RGB')

            c_img = np.array(c_img1)

            

            c_mask1 = np.expand_dims(imread(biaoqian_path), -1)

            c_mask = np.array(c_mask1,dtype=np.float32)

            

            if IMG_SCALING is not None:

                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]

                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]

            out_rgb += [c_img]

            out_mask += [c_mask]

            if len(out_rgb)>=batch_size:

                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)

                out_rgb, out_mask=[], []
valid_x, valid_y = next(make_image_gen(valid_df, VALID_IMG_COUNT))

print(valid_x.shape, valid_y.shape)
from keras.preprocessing.image import ImageDataGenerator

dg_args = dict(featurewise_center = False, 

                  samplewise_center = False,

                  rotation_range = 45, 

                  width_shift_range = 0.1, 

                  height_shift_range = 0.1, 

                  shear_range = 0.01,

                  zoom_range = [0.9, 1.25],  

                  horizontal_flip = True, 

                  vertical_flip = True,

                  fill_mode = 'reflect',

                   data_format = 'channels_last')

# brightness can be problematic since it seems to change the labels differently from the images 

#亮度可能有问题，因为它似乎改变了不同于图像的标签

if AUGMENT_BRIGHTNESS:

    dg_args[' brightness_range'] = [0.5, 1.5]

image_gen = ImageDataGenerator(**dg_args)

#**kwargs 表示关键字参数，它本质上是一个 dict

if AUGMENT_BRIGHTNESS:

    dg_args.pop('brightness_range')

label_gen = ImageDataGenerator(**dg_args)

#pop删除 arrayObject 的最后一个元素



def create_aug_gen(in_gen, seed = None):

    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    for in_x, in_y in in_gen:

        seed = np.random.choice(range(9999))

        ##保持种子同步否则对图像的增强与遮罩不同

        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks

        g_x = image_gen.flow(255*in_x, 

                             batch_size = in_x.shape[0], 

                             seed = seed, 

                             shuffle=True)

        g_y = label_gen.flow(in_y, 

                             batch_size = in_x.shape[0], 

                             seed = seed, 

                             shuffle=True)



        yield next(g_x)/255.0, next(g_y)
gc.collect()
from keras.preprocessing import image

from keras.models import Model

from keras import layers

from keras.layers import Activation, AveragePooling2D, BatchNormalization, Concatenate

from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Lambda, MaxPooling2D

from keras.layers import SeparableConv2D, DepthwiseConv2D

from keras.layers import Add, Multiply, Reshape

from keras.applications.imagenet_utils import decode_predictions

from keras.utils.data_utils import get_file

from keras import backend as K



from keras.utils.generic_utils import get_custom_objects





def relu6(x):

    # relu函数

    return K.relu(x, max_value=6.0)





get_custom_objects().update({'relu6': Activation(relu6)})





def hard_swish(x):

    # 利用relu函数乘上x模拟sigmoid

    return x * K.relu(x + 3.0, max_value=6.0) / 6.0





get_custom_objects().update({'hard_swish': Activation(hard_swish)})





def return_activation(x, nl):

    # 用于判断使用哪个激活函数

    if nl == 'HS':

        x = Activation(hard_swish)(x)

    if nl == 'RE':

        x = Activation(relu6)(x)

    return x





def channel_split(x, name=''):

    in_channels = x.shape.as_list()[-1]

    ip = in_channels // 2

    c_hat = Lambda(lambda z: z[:, :, :, 0:ip])(x)

    c = Lambda(lambda z: z[:, :, :, ip:])(x)



    return c_hat, c





def channel_shuffle(x):

    height, width, channels = x.shape.as_list()[1:]

    channels_per_split = channels // 2



    x = K.reshape(x, [-1, height, width, 2, channels_per_split])

    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))

    x = K.reshape(x, [-1, height, width, channels])



    return x





def squeeze(inputs):

    # 注意力机制单元

    input_channels = int(inputs.shape[-1])



    x = GlobalAveragePooling2D()(inputs)

    x = Dense(int(input_channels / 4))(x)

    x = Activation(relu6)(x)

    x = Dense(input_channels)(x)

    x = Activation(hard_swish)(x)

    x = Reshape((1, 1, input_channels))(x)

    x = Multiply()([inputs, x])



    return x





def _shuffle_unit(inputs, out_channels, sq, nl, strides=2, stage=1, block=1):

    bn_axis = -1  # 通道在后还是在前

    prefix = 'stage%d/block%d' % (stage, block)



    branch_channels = out_channels // 2



    if strides == 2:

        x_1 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',

                              use_bias=False, name='%s/3x3dwconv_1' % prefix)(inputs)

        x_1 = BatchNormalization(axis=bn_axis, name='%s/bn_3x3dwconv_1' % prefix)(x_1)

        x_1 = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',

                     use_bias=False, name='%s/1x1conv_1' % prefix)(x_1)

        x_1 = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_1' % prefix)(x_1)

        x_1 = Activation('relu6')(x_1)



        x_2 = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',

                     use_bias=False, name='%s/1x1conv_2' % prefix)(inputs)

        x_2 = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_2' % prefix)(x_2)

        x_2 = Activation('relu6')(x_2)

        x_2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',

                              use_bias=False, name='%s/3x3dwconv_2' % prefix)(x_2)

        x_2 = BatchNormalization(axis=bn_axis, name='%s/bn_3x3dwconv_2' % prefix)(x_2)

        x_2 = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',

                     use_bias=False, name='%s/1x1conv_3' % prefix)(x_2)

        x_2 = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_3' % prefix)(x_2)

        x_2 = Activation('relu6')(x_2)



        x = Concatenate(axis=bn_axis, name='%s/concat' % prefix)([x_1, x_2])



    if strides == 1:

        c_hat, c = channel_split(inputs, name='%s/split' % prefix)



        c = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',

                   use_bias=False, name='%s/1x1conv_4' % prefix)(c)

        # c = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_4' % prefix)(c)

        # c = Activation('relu6')(c)

        c = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',

                            use_bias=False, name='%s/3x3dwconv_3' % prefix)(c)

        c = BatchNormalization(axis=bn_axis, name='%s/bn_3x3dwconv_3' % prefix)(c)

        # c = Activation('relu6')(c)

        c = return_activation(c, nl)

        # 引入注意力机制

        if sq:

            c = squeeze(c)

        # 下降通道数

        c = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',

                   use_bias=False, name='%s/1x1conv_5' % prefix)(c)

        c = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_4' % prefix)(c)

        x = Concatenate(axis=bn_axis, name='%s/concat' % prefix)([c_hat, c])



    x = Lambda(channel_shuffle, name='%s/channel_shuffle' % prefix)(x)



    return x





def exblock(inputs, out_channels, sq, stage=1, block=1):

    prefix = 'stage%d/block%d' % (stage, block)



    residual = Conv2D(out_channels, (1, 1), strides=(2, 2), padding='same', use_bias=False)(inputs)

    residual = BatchNormalization()(residual)



    x = SeparableConv2D(out_channels, (3, 3), padding='same', use_bias=False, name='%s/_sepconv1' % prefix)(inputs)

    x = BatchNormalization(name='%s/_sepconv1_bn' % prefix)(x)

    x = Activation('hard_swish', name='%s/_sepconv2_ac_hs' % prefix)(x)

    x = SeparableConv2D(out_channels, (3, 3), padding='same', use_bias=False, name='%s/_sepconv2' % prefix)(x)

    # 引入注意力机制

    if sq:

        x = squeeze(x)



    x = BatchNormalization(name='%s/_sepconv2_bn' % prefix)(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='%s/_pool' % prefix)(x)

    x = layers.add([x, residual])



    return x





def inception_unit(inputs, channel1, channel2, channel3, ):

    branch_0 = Conv2D(channel1, (1, 1), strides=(1, 1), padding='same', use_bias=False)(inputs)

    branch_0 = BatchNormalization(axis=-1, scale=False, name='stage1X1_1BN')(branch_0)

    branch_0 = Activation('relu6', name='stage1X1_1ac')(branch_0)



    branch_1 = Conv2D(channel2, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)

    branch_1 = BatchNormalization(axis=-1, scale=False, name='stage3X3_1BN')(branch_1)

    branch_1 = Activation('relu6', name='stage3X3_1ac')(branch_1)



    branch_pool = AveragePooling2D(3, strides=1, padding='same')(inputs)

    branch_pool = Conv2D(channel3, (1, 1), strides=(1, 1), padding='same', use_bias=False)(branch_pool)

    branch_pool = BatchNormalization(axis=-1, scale=False, name='stagep1X1_1BN')(branch_pool)

    branch_pool = Activation('relu6', name='stagep1X1_1ac')(branch_pool)



    branches = [branch_0, branch_1, branch_pool]



    x = Concatenate(name='mixed_5b')(branches)



    return x





def qzynetnew(input_shape=[256, 320, 3], classes=2,target=1):

    input_shape = [256, 320, 3]



    img_input = Input(shape=input_shape)



    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(img_input)

    x = BatchNormalization(axis=-1, scale=False, name='stage0.1X1_1BN')(x)

    x = Activation('relu6', name='stage0.1X1_1ac')(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)

    x = BatchNormalization(axis=-1, scale=False, name='stage00.1X1_1BN')(x)

    x = Activation('relu6', name='stage00.1X1_1ac')(x)

    x = MaxPooling2D(3, strides=1,padding='same')(x)#2

    #   x=_shuffle_unit(x, 128, sq=False, nl='RE',strides=1, stage=2, block=1)

    #   x=_shuffle_unit(x, 128, sq=False, nl='RE',strides=1, stage=2, block=2)



    #   x=_shuffle_unit(x, 128, sq=False, nl='RE',strides=2, stage=2, block=3)#128,128,128 -> 64 x 64 x 128



    #   x=_shuffle_unit(x, 128, sq=False, nl='RE',strides=1, stage=2, block=4)

    #   x=_shuffle_unit(x, 128, sq=False, nl='RE',strides=1, stage=2, block=5)

    f1=x

    x = exblock(x, 128, sq=True, stage=1, block=1)

    f2=x



    x = exblock(x, 192, sq=True, stage=1, block=2)

    # x=_shuffle_unit(x, 256, sq=False, nl='RE',strides=2, stage=2, block=6)#64,64,128 -> 32 x 32 x 256

    x = inception_unit(x, 116, 116, 24)

    x = _shuffle_unit(x, 256, sq=False, nl='RE', strides=1, stage=2, block=7)

    x = _shuffle_unit(x, 256, sq=False, nl='RE', strides=1, stage=2, block=8)

    f3= x



    x = _shuffle_unit(x, 512, sq=False, nl='RE', strides=2, stage=2, block=9)  # 32,32,256 -> 16 x 16 x 512



    x = _shuffle_unit(x, 512, sq=False, nl='RE', strides=1, stage=2, block=10)

    x = _shuffle_unit(x, 512, sq=False, nl='RE', strides=1, stage=2, block=11)

    x = _shuffle_unit(x, 512, sq=False, nl='RE', strides=1, stage=2, block=12)

    f4= x



    x = _shuffle_unit(x, 1024, sq=False, nl='RE', strides=2, stage=2, block=13)  # 16 x 16 x 512 -> 8 x 8 x 1024



    x = _shuffle_unit(x, 1024, sq=False, nl='RE', strides=1, stage=2, block=14)

    f5= x



    if target == 1:

         x = GlobalAveragePooling2D(name='global_max_pool')(x)

         x = Dense(classes, name='fc')(x)

         x = Activation('softmax')(x)



         inputs = img_input

    # 创建模型

         model = Model(inputs, x, name='qzynet')

         return model



    if target == 2:

         return img_input, [f1, f2, f3, f4, f5]
from keras.preprocessing import image

from keras.models import Model

from keras import layers

from keras.layers import Activation, AveragePooling2D, BatchNormalization, Concatenate

from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Lambda, MaxPooling2D

from keras.layers import SeparableConv2D, DepthwiseConv2D

from keras.layers import Add, Multiply, Reshape

from keras.layers import ZeroPadding2D, UpSampling2D, concatenate

from keras.applications.imagenet_utils import decode_predictions

from keras.utils.data_utils import get_file

from keras import backend as K



# from qzynetwork1 import qzynetnew



IMAGE_ORDERING = 'channels_last'

MERGE_AXIS = -1



def conv_block(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1, w_init='he_normal'):

    x = (Conv2D(filters=num_filters,

                               kernel_size=kernel_size,

                               padding=padding,

                               strides=strides,

                               dilation_rate=dilation_rate,

                               kernel_initializer=w_init,

                               use_bias=False))(tensor)

    x = (BatchNormalization())(x)

    x =  Activation('relu')(x)



    return x





def sepconv_block(tensor, num_filters, kernel_size, padding='same', strides=1, dilation_rate=1, w_init='he_normal'):

    x = (SeparableConv2D(filters=num_filters,

                                        depth_multiplier=1,

                                        kernel_size=kernel_size,

                                        padding=padding,

                                        strides=strides,

                                        dilation_rate=dilation_rate,

                                        depthwise_initializer=w_init,

                                        use_bias=False))(tensor)

    x =(BatchNormalization())(x)

    x = Activation('relu')(x)

    return x





# def JPU(encoder=qzynetnew, out_channels=512):

#     img_inputs, levels = encoder(input_shape=[256, 256, 3], classes=2,target=2)

#     [f2, f3, f4, f5] = levels  # f5:8,f4:16,f3:32,f2:64

#     #h=128

#     #w=128

#

#     # yc = UpSampling2D(size=(2, 2), interpolation='bilinear')(yc)#得到128

#     # for i in range(1, 4):

#     #     levels[i] = conv_block(levels[i], out_channels, 3)

#     #     if i != 1:

#     #         h_t, w_t = levels[i].shape.as_list()[1:3]

#     #         scale = (h // h_t, w // w_t)

#     #         levels[i] = tf.keras.layers.UpSampling2D(

#     #             size=scale, interpolation='bilinear')(levels[i])

#     # yc = tf.keras.layers.Concatenate(axis=-1)(levels[1:])

#     ym = []

#     for rate in [1, 2]:

#         ym.append(sepconv_block(yc, 512, 3, dilation_rate=rate))

#     y = Concatenate(axis=-1)(ym)

#

#     y = conv_block(y, num_filters=128, kernel_size=1)

#     # return  y

#     model = Model(img_inputs,y,name='jpu')

#

#     return model



def _unet(n_classes=2, encoder=qzynetnew,  input_height=256, input_width=320):

    img_input, levels = encoder(input_shape=[256, 320, 3], classes=2,target=2)

    [f1, f2, f3, f4, f5] = levels#f5:8,f4:16,f3:32,f2:64



    o = f5

    # 8,8,512

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)

    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)

    o = (BatchNormalization())(o)



    # 16,16,512

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    # 16,16,768

    o = (concatenate([o, f4], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)

    # 16,16,256

    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)

    o = (BatchNormalization())(o)



    # 32,32,256

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    # 32,32,384

    o = (concatenate([o, f3], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)

    # 32,32,128

    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)

    o = (BatchNormalization())(o)

    # 64,64,64

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)



    o = (concatenate([o, f2], axis=MERGE_AXIS))



    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)

    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)

    o = (BatchNormalization())(o)



    f5 = conv_block(f5, 96, 3)#8

    f4 = conv_block(f4, 64, 3)#16

    f3 = conv_block(f3, 32, 3)#32

    f5 = UpSampling2D(size=(8, 8))(f5)

    f4 = UpSampling2D(size=(4, 4))(f4)

    f3 = UpSampling2D(size=(2, 2))(f3)

    yc = Concatenate(axis=-1)([f3,f4,f5])

    ym = []

    for rate in [1, 2]:

        ym.append(sepconv_block(yc, 64, 3, dilation_rate=rate))

    y = Concatenate(axis=-1)(ym)

    y = conv_block(y, num_filters=64, kernel_size=1)



    z = concatenate([o, y])

    z = (Conv2D(64, (3, 3), padding='same'))(z)#得到64*64*32

    z = (BatchNormalization())(z)



    z = UpSampling2D(size=(2,2))(z)



    

    z = (concatenate([z, f1], axis=MERGE_AXIS))

    z = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(z)

    z = (Conv2D(32, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(z)

    z = (BatchNormalization())(z)

    z = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(z)

    

    z = (Conv2D(1, (1, 1), padding='same'))(z)

    z = Activation('sigmoid')(z)

    model = Model(inputs=img_input, outputs=z, name='jpu_unet')



    return model
seg_model=_unet()

seg_model.summary()
import keras.backend as K

import tensorflow as tf

from keras.optimizers import Adam



def IoU(y_true, y_pred, eps=1e-6):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])

    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection

    return K.mean( (intersection + eps) / (union + eps), axis=0)



def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred = K.cast(y_pred, 'float32')

    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')

    intersection = y_true_f * y_pred_f

    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

    return score



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score





gamma=2.0,

alpha=0.25



def focal_loss(y_true, y_pred):

        # Define epsilon so that the backpropagation will not result in NaN

        # for 0 divisor case

    epsilon = K.epsilon()

        # Add the epsilon to prediction value

        # y_pred = y_pred + epsilon

        # Clip the prediciton value

    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate p_t

    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)

        # Calculate alpha_t

    alpha_factor = K.ones_like(y_true) * alpha

    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)

        # Calculate cross entropy

    cross_entropy = -K.log(p_t)

    weight = alpha_t * K.pow((1 - p_t), gamma)

        # Calculate focal loss

    loss = weight * cross_entropy

        # Sum the losses in mini_batch

    loss = K.sum(loss, axis=1)

    return loss
def IoUloss(y_true, y_pred, eps=1e-6):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])

    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection

    return 1-K.mean( (intersection + eps) / (union + eps), axis=0)
def mixed(y_true, y_pred):

    loss =focal_loss(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

    return loss
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('seg_model')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

#weight_path保存模型的路径，monitor：需要监视的值，verbose：信息展示模式，save_best_only：当设置为True时，监测值有改进时才会保存当前的模型

#save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.6,

                                   patience=2, verbose=1, mode='min',

                                   min_delta=0.0001, cooldown=0, min_lr=1e-8)



#当指标停止提升时，降低学习速率。

#monitor：要监测的数量。patience：没有提升的epoch数，之后学习率将降低。verbose：int。0：安静，1：更新消息。

#mode：{auto，min，max}之一。在min模式下，当监测量停止下降时，lr将减少；在max模式下，当监测数量停止增加时，它将减少；

#在auto模式下，从监测数量的名称自动推断方向。

#min_delta：对于测量新的最优化的阀值，仅关注重大变化。

#cooldown：在学习速率被降低之后，重新恢复正常操作之前等待的epoch数量。

#min_lr：学习率的下限。





early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,

                      patience=10) # probably needs to be more patient, but kaggle time is limited

#目的：防止过拟合

#monitor: 需要监视的量，val_loss，val_acc

#patience: 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练

#verbose: 信息展示模式

#mode: 'auto','min','max'之一，在min模式训练，如果检测值停止下降则终止训练。在max模式下，当检测值不再上升的时候则停止训练。



callbacks_list = [checkpoint, early, reduceLROnPlat]
MAX_TRAIN_EPOCHS=40

BATCH_SIZE=28

MAX_TRAIN_STEPS=300
a=train_df.shape[0]//BATCH_SIZE

print(a)
seg_model.compile(optimizer=Adam(1e-2, decay=1e-6), loss=IoUloss, metrics=[IoU,dice_coef,'binary_accuracy'])#tversky_loss



step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)

aug_gen = create_aug_gen(make_image_gen(train_df))

loss_history = [seg_model.fit_generator(aug_gen,

                                 steps_per_epoch=step_count,

                                 epochs=MAX_TRAIN_EPOCHS,

                                 validation_data=(valid_x, valid_y),

                                 callbacks=callbacks_list,

                                workers=1 # the generator is not very thread safe

                                           )]
def show_loss(loss_history):

    epochs = np.concatenate([mh.epoch for mh in loss_history])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    

    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',

                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')

    ax1.legend(['Training', 'Validation'])

    ax1.set_title('Loss')

    

    _ = ax2.plot(epochs, np.concatenate([mh.history['IoU'] for mh in loss_history]), 'b-',

                 epochs, np.concatenate([mh.history['val_IoU'] for mh in loss_history]), 'r-')

    ax2.legend(['Training', 'Validation'])

    ax2.set_title('IoU Accuracy (%)')



show_loss(loss_history)
seg_model.load_weights(weight_path)

seg_model.save_weights('seg_model111.h5')
def predict(img, path=train_image_dir):

    

    c_img=Image.open(os.path.join(path, c_img_name)).convert('RGB')

    c_img = np.array(c_img)

    

#     c_img = imread(os.path.join(path, c_img_name))

    c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]

    c_img = np.expand_dims(c_img, 0)/255.0

    

    cur_seg = seg_model.predict(c_img)[0]

    cur_seg = binary_opening(cur_seg>0.6, np.expand_dims(disk(2), -1))

    return cur_seg, c_img



def pred_encode(img):

    cur_seg, _ = predict(img)

    cur_rles = rle_encode(cur_seg)

    return [img, cur_rles if len(cur_rles) > 0 else None]



## Get a sample of each group of ship count

samples = train_df.sample(n=10)



fig, m_axs = plt.subplots(samples.shape[0], 3, figsize = (11, samples.shape[0]*4))

[c_ax.axis('off') for c_ax in m_axs.flatten()]



for (ax1, ax2, ax3), c_img_name in zip(m_axs, samples.ImageId.values):

    first_seg, first_img = predict(c_img_name, train_image_dir)

    ax1.imshow(first_img[0])

    ax1.set_title('Image')

    ax2.imshow(first_seg[:, :, 0])

    ax2.set_title('Prediction')

    ground_truth = imread(os.path.join(mask_image_dir, c_img_name))

    ax3.imshow( ground_truth)

    ax3.set_title('ground_truth')

    

fig.savefig('predictions.png')