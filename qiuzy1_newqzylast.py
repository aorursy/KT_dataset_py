BATCH_SIZE = 36

EDGE_CROP = 16

GAUSSIAN_NOISE = 0.1

UPSAMPLE_MODE = 'SIMPLE'

# downsampling inside the network

NET_SCALING = (1, 1)

# downsampling in preprocessing

IMG_SCALING = (3, 3)

# number of validation images to use

VALID_IMG_COUNT = 600

# maximum number of steps_per_epoch in training

MAX_TRAIN_STEPS = 1000#每次迭代样本数

AUGMENT_BRIGHTNESS = False
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread

import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

from skimage.util import montage

from skimage.morphology import binary_opening, disk



montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

ship_dir = '../input/airbus-ship-detection'

train_image_dir = os.path.join(ship_dir, 'train_v2')

test_image_dir = os.path.join(ship_dir, 'test_v2')

import gc; gc.enable() # memory is tight



from skimage.morphology import label

def multi_rle_encode(img):

    labels = label(img)

    if img.ndim > 2:

        return [rle_encode(np.sum(labels==k, axis=2)) for k in np.unique(labels[labels>0])]

    else:

        return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]



# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

def rle_encode(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    if np.max(img) < 1e-3:

        return '' ## no need to encode if it's all zeros

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle_decode(mask_rle, shape=(768, 768)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background

    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction



def masks_as_image(in_mask_list):

    # Take the individual ship masks and create a single mask array for all ships

    all_masks = np.zeros((768, 768), dtype = np.uint8)

    for mask in in_mask_list:

        if isinstance(mask, str):

            all_masks |= rle_decode(mask)

    return all_masks



def masks_as_color(in_mask_list):

    # Take the individual ship masks and create a color mask array for each ships

    all_masks = np.zeros((768, 768), dtype = np.float)

    channel = 0 ## alternate through color channels

    for i,mask in enumerate(in_mask_list):

        if isinstance(mask, str):

            all_masks[:,:] += np.log1p(i+1) * rle_decode(mask)

    return all_masks
ship_dir1 = '../input/airbus-ship-detection'

masks = pd.read_csv(os.path.join(ship_dir1, 'train_ship_segmentations_v2.csv'))

not_empty = pd.notna(masks.EncodedPixels)

print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')

print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')

masks.head()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (16, 5))

rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']

img_0 = masks_as_image(rle_0)

ax1.imshow(img_0)

ax1.set_title('Mask as image')

rle_1 = multi_rle_encode(img_0)

img_1 = masks_as_image(rle_1)

ax2.imshow(img_1)

ax2.set_title('Re-encoded')

img_c = masks_as_color(rle_0)

ax3.imshow(img_c)

ax3.set_title('Masks in colors')

img_c = masks_as_color(rle_1)

ax4.imshow(img_c)

ax4.set_title('Re-encoded in colors')

print('Check Decoding->Encoding',

      'RLE_0:', len(rle_0), '->',

      'RLE_1:', len(rle_1))

print(np.sum(img_0 - img_1), 'error')
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()

unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)

unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

# some files are too small/corrupt

unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: 

                                                               os.stat(os.path.join(train_image_dir, 

                                                                                    c_img_id)).st_size/1024)

unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50] # keep only +50kb files

unique_img_ids['file_size_kb'].hist()

masks.drop(['ships'], axis=1, inplace=True)

unique_img_ids.sample(7)
unique_img_ids= unique_img_ids[unique_img_ids['ships']!=0]
SAMPLES_PER_GROUP = 1800

# unique_img_ids['grouped_ship_count'] = unique_img_ids['ships'].map(lambda x: (x+2)//3)

# balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

# balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)

# print(balanced_train_df.shape[0], 'masks')

balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

#图片有相同船舶数量，但超出2000的不要

rect=plt.hist(x = balanced_train_df['ships'], # 指定绘图数据

           bins = 15, # 指定直方图中条块的个数

           color = 'steelblue', # 指定直方图的填充色

           edgecolor = 'black' # 指定直方图的边框色

          )

plt.yticks(range(0,1800,300))#1800

plt.xticks(range(0,14))

plt.ylabel("Number of images")

plt.xlabel('Number of ships')

plt.title("Number of images containing different number of vessels")

#balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)

print(balanced_train_df.shape[0], 'images',balanced_train_df.shape)#取出1万张图片
balanced_train_df=balanced_train_df.reset_index(drop = True)#删除原来的索引。

balanced_train_df=balanced_train_df.sample(frac=1.0)

balanced_train_df.head()
from sklearn.model_selection import train_test_split

train_ids, valid_ids = train_test_split(balanced_train_df, 

                 test_size = 0.3, 

                 stratify = balanced_train_df['ships'])

train_df = pd.merge(masks, train_ids)

valid_df = pd.merge(masks, valid_ids)

print(train_df.shape[0], 'training masks')

print(valid_df.shape[0], 'validation masks')
def make_image_gen(in_df, batch_size = BATCH_SIZE):

    all_batches = list(in_df.groupby('ImageId'))

    out_rgb = []

    out_mask = []

    while True:

        np.random.shuffle(all_batches)

        for c_img_id, c_masks in all_batches:

            rgb_path = os.path.join(train_image_dir, c_img_id)

            c_img = imread(rgb_path)

            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)

            if IMG_SCALING is not None:

                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]

                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]

            out_rgb += [c_img]

            out_mask += [c_mask]

            if len(out_rgb)>=batch_size:

                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)

                out_rgb, out_mask=[], []
train_gen = make_image_gen(train_df)

train_x, train_y = next(train_gen)

print('x', train_x.shape, train_x.min(), train_x.max())

print('y', train_y.shape, train_y.min(), train_y.max())
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))

batch_rgb = montage_rgb(train_x)

batch_seg = montage(train_y[:, :, :, 0])

ax1.imshow(batch_rgb)

ax1.set_title('Images')

ax2.imshow(batch_seg)

ax2.set_title('Segmentations')

ax3.imshow(mark_boundaries(batch_rgb, 

                           batch_seg.astype(int)))

ax3.set_title('Outlined Ships')

fig.savefig('overview.png')
%%time

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

if AUGMENT_BRIGHTNESS:

    dg_args[' brightness_range'] = [0.5, 1.5]

image_gen = ImageDataGenerator(**dg_args)



if AUGMENT_BRIGHTNESS:

    dg_args.pop('brightness_range')

label_gen = ImageDataGenerator(**dg_args)



def create_aug_gen(in_gen, seed = None):

    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))

    for in_x, in_y in in_gen:

        seed = np.random.choice(range(9999))

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
cur_gen = create_aug_gen(train_gen)

t_x, t_y = next(cur_gen)

print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())

print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())

# only keep first 9 samples to examine in detail

t_x = t_x[:9]

t_y = t_y[:9]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

ax1.imshow(montage_rgb(t_x), cmap='gray')

ax1.set_title('images')

ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray_r')

ax2.set_title('ships')
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





def qzynetnew(input_shape=[256, 256, 3], classes=2,target=1):

    input_shape = [256, 256, 3]



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



    x = _shuffle_unit(x, 1024, sq=True, nl='RE', strides=2, stage=2, block=13)  # 16 x 16 x 512 -> 8 x 8 x 1024



    x = _shuffle_unit(x, 1024, sq=True, nl='HS', strides=1, stage=2, block=14)

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

         return img_input, [f2, f3, f4, f5]
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



def _unet(n_classes=2, encoder=qzynetnew,  input_height=256, input_width=256):

    img_input, levels = encoder(input_shape=[256, 256, 3], classes=2,target=2)

    [f2, f3, f4, f5] = levels#f5:8,f4:16,f3:32,f2:64



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

    f5 = UpSampling2D(size=(8, 8), interpolation='bilinear')(f5)

    f4 = UpSampling2D(size=(4, 4), interpolation='bilinear')(f4)

    f3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(f3)

    yc = Concatenate(axis=-1)([f3,f4,f5])

    ym = []

    for rate in [1, 2]:

        ym.append(sepconv_block(yc, 64, 3, dilation_rate=rate))

    y = Concatenate(axis=-1)(ym)

    y = conv_block(y, num_filters=128, kernel_size=1)



    z = concatenate([o, y])

    z = (Conv2D(32, (3, 3), padding='same'))(z)#得到64*64*32

    z = (BatchNormalization())(z)



    z = UpSampling2D(size=(4,4), interpolation='bilinear')(z)

    z = (Conv2D(1, (1, 1), padding='same'))(z)

    model = Model(input=img_input, output=z, name='jpu_unet')



    return model
seg_model=_unet()
def dice_loss(input, target):

    input = K.sigmoid(input)

    smooth = 1.0



#     input=array(input,'f')

#     target=array(target,'f')

    

    iflat = K.flatten(input)

    tflat = K.flatten(target)

    intersection = K.sum((iflat * tflat))

    

    return ((2.0 * intersection + smooth) / (K.sum(iflat)+ K.sum(tflat) + smooth))
from keras import backend as K

import tensorflow as tf



def KerasFocalLoss(input,target):

    

    gamma = 2.

    input = tf.cast(input, tf.float32)

    

    max_val = K.clip(-input, 0, 1)

    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))

    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))

    loss = K.exp(invprobs * gamma) * loss

    

    loss1=K.mean(K.sum(loss, axis=1))

    return loss1
def KerasFocalLoss1(input,target):

    

    gamma = 2.

    input = tf.cast(input, tf.float32)

    

    max_val = K.clip(-input, 0, 1)

    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))

    invprobs = tf.math.log_sigmoid(-input * (target * 2.0 - 1.0))

    loss = K.exp(invprobs * gamma) * loss

    

    loss1=K.mean(loss)

    return loss1
def mixedLoss(y_true,y_pred):

    alpha=10

    loss=K.mean(alpha * KerasFocalLoss1(y_true,y_pred) - K.log(dice_loss(y_true,y_pred)))

    return loss
import keras.backend as K

from keras.optimizers import Adam

from keras.losses import binary_crossentropy



## IoU of boats

def IoU(y_true, y_pred, eps=1e-6):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])

    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection

    return K.mean( (intersection + eps) / (union + eps), axis=0)



## IoU of non-boats

def zero_IoU(y_true, y_pred):

    return IoU(1-y_true, 1-y_pred)



def agg_loss(in_gt, in_pred):

    return -1e-2 * zero_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)



seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss='binary_crossentropy', metrics=[IoU, zero_IoU, 'binary_accuracy'])
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('seg_model')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,

                             save_best_only=True, mode='min', save_weights_only = True)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                                   patience=1, verbose=1, mode='min',

                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)



early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,

                      patience=15) # probably needs to be more patient, but kaggle time is limited



callbacks_list = [checkpoint, early, reduceLROnPlat]
step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)

aug_gen = create_aug_gen(make_image_gen(train_df))

loss_history = [seg_model.fit_generator(aug_gen, 

                             steps_per_epoch=step_count, 

                             epochs=10, 

                             validation_data=(valid_x, valid_y),

                             callbacks=callbacks_list,

                            workers=1, # the generator is not very thread safe,

                            max_queue_size = 20,use_multiprocessing=True,verbose=1)]
def show_loss(loss_history):

    epich = np.cumsum(np.concatenate(

        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))

    _ = ax1.plot(epich,

                 np.concatenate([mh.history['loss'] for mh in loss_history]),

                 'b-',

                 epich, np.concatenate(

            [mh.history['val_loss'] for mh in loss_history]), 'r-')

    ax1.legend(['Training', 'Validation'])

    ax1.set_title('Loss')

    

    _ = ax2.plot(epich, np.concatenate(

        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',

                     epich, np.concatenate(

            [mh.history['val_binary_accuracy'] for mh in loss_history]),

                     'r-')

    ax2.legend(['Training', 'Validation'])

    ax2.set_title('Binary Accuracy (%)')

    

    _ = ax3.plot(epich, np.concatenate(

        [mh.history['IoU'] for mh in loss_history]), 'b-',

                     epich, np.concatenate(

            [mh.history['val_IoU'] for mh in loss_history]),

                     'r-')

    ax3.legend(['Training', 'Validation'])

    ax3.set_title('Boat IoU (%)')

    

    _ = ax4.plot(epich, np.concatenate(

        [mh.history['zero_IoU'] for mh in loss_history]), 'b-',

                     epich, np.concatenate(

            [mh.history['val_zero_IoU'] for mh in loss_history]),

                     'r-')

    ax4.legend(['Training', 'Validation'])

    ax4.set_title('Non-boat IoU')



show_loss(loss_history)
seg_model.load_weights(weight_path)

seg_model.save_weights('seg_model111.h5')
pred_y = seg_model.predict(valid_x)

print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())
fig, ax = plt.subplots(1, 1, figsize = (6, 6))

ax.hist(pred_y.ravel(), np.linspace(0, 1, 10))

ax.set_xlim(0, 1)

ax.set_yscale('log', nonposy='clip')
if IMG_SCALING is not None:

    fullres_model = models.Sequential()

    fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))

    fullres_model.add(seg_model)

    fullres_model.add(layers.UpSampling2D(IMG_SCALING))

else:

    fullres_model = seg_model

fullres_model.save('fullres_model.h5')
def predict(img, path=test_image_dir):

    c_img = imread(os.path.join(path, c_img_name))

    c_img = np.expand_dims(c_img, 0)/255.0

    cur_seg = fullres_model.predict(c_img)[0]

    cur_seg = binary_opening(cur_seg>1e3, np.expand_dims(disk(2), -1))

    return cur_seg, c_img



def pred_encode(img):

    cur_seg, _ = predict(img)

    cur_rles = rle_encode(cur_seg)

    return [img, cur_rles if len(cur_rles) > 0 else None]



## Get a sample of each group of ship count

samples = train_df.groupby('grouped_ship_count').apply(lambda x: x.sample(1))

fig, m_axs = plt.subplots(samples.shape[0], 3, figsize = (11, samples.shape[0]*4))

[c_ax.axis('off') for c_ax in m_axs.flatten()]



for (ax1, ax2, ax3), c_img_name in zip(m_axs, samples.ImageId.values):

    first_seg, first_img = predict(c_img_name, train_image_dir)

    ax1.imshow(first_img[0])

    ax1.set_title('Image')

    ax2.imshow(first_seg[:, :, 0])

    ax2.set_title('Prediction')

    ground_truth = masks_as_color(masks.query('ImageId=="{}"'.format(c_img_name))['EncodedPixels'])

    ax3.imshow(ground_truth)

    ax3.set_title('Ground Truth')

    

fig.savefig('predictions.png')
test_paths = np.array(os.listdir(test_image_dir))

print(len(test_paths), 'test images found')
%%time

from tqdm import tqdm_notebook



out_pred_rows = []

for c_img_name in tqdm_notebook(test_paths[:30000]): ## only a subset as it takes too long to run

    out_pred_rows += [pred_encode(c_img_name)]
sub = pd.DataFrame(out_pred_rows)

sub.columns = ['ImageId', 'EncodedPixels']

sub = sub[sub.EncodedPixels.notnull()]

sub.head()
sub1 = pd.read_csv('../input/airbus-ship-detection/sample_submission_v2.csv')

sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])

sub1['EncodedPixels'] = None

print(len(sub1), len(sub))



sub = pd.concat([sub, sub1])

print(len(sub))

sub.to_csv('submission.csv', index=False)

sub.head()