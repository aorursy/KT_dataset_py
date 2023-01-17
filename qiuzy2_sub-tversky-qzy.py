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



def _unet(n_classes=2, encoder=qzynetnew,  input_height=256, input_width=256):

    img_input, levels = encoder(input_shape=[256, 256, 3], classes=2,target=2)

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

    model = Model(input=img_input, output=z, name='jpu_unet')



    return model



seg_model =_unet()

seg_model.load_weights("../input/qzy-tversky-40/seg_model_weights.best.hdf5")

seg_model.summary()
from skimage.io import imread

import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap

from skimage.segmentation import mark_boundaries

from skimage.util import montage

from skimage.morphology import binary_opening, disk, label

import gc; gc.enable()



test_image_dir='../input/airbus-ship-detection/test_v2'
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

ship_dir = '../input'





def multi_rle_encode(img, **kwargs):

    '''

    Encode connected regions as separated masks

    '''

    labels = label(img)

    if img.ndim > 2:

        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]

    else:

        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]



# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    if np.max(img) < min_max_threshold:

        return '' ## no need to encode if it's all zeros

    if max_mean_threshold and np.mean(img) > max_mean_threshold:

        return '' ## ignore overfilled mask

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

    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 

    for i,mask in enumerate(in_mask_list):

        if isinstance(mask, str):

            all_masks[:,:] += scale(i) * rle_decode(mask)

    return all_masks
import os

import numpy as np  

test_paths = np.array(os.listdir('../input/airbus-ship-detection/test_v2'))

print(len(test_paths), 'test images found')
IMG_SCALING=(3,3)
def raw_prediction(img, path=test_image_dir):

    c_img = imread(os.path.join(path, c_img_name))

    c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]

    c_img = np.expand_dims(c_img, 0)/255.0

    cur_seg = seg_model.predict(c_img)[0]

    return cur_seg, c_img[0]



def smooth(cur_seg):

    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))



def predict(img, path=test_image_dir):

    cur_seg, c_img = raw_prediction(img, path=path)

    return smooth(cur_seg), c_img



from tqdm import tqdm_notebook



def pred_encode(img, **kwargs):

    cur_seg, _ = predict(img)

    cur_rles = multi_rle_encode(cur_seg, **kwargs)

    return [[img, rle] for rle in cur_rles if rle is not None]



out_pred_rows = []

for c_img_name in tqdm_notebook(test_paths[:30000]): ## only a subset as it takes too long to run

    out_pred_rows += pred_encode(c_img_name, min_max_threshold=1.0)



import pandas as pd 



sub = pd.DataFrame(out_pred_rows)

sub.columns = ['ImageId', 'EncodedPixels']

sub = sub[sub.EncodedPixels.notnull()]

sub.head()
TOP_PREDICTIONS=5

fig, m_axs = plt.subplots(TOP_PREDICTIONS, 2, figsize = (9, TOP_PREDICTIONS*5))

[c_ax.axis('off') for c_ax in m_axs.flatten()]



for (ax1, ax2), c_img_name in zip(m_axs, sub.ImageId.unique()[:TOP_PREDICTIONS]):

    c_img = imread(os.path.join(test_image_dir, c_img_name))

    c_img = np.expand_dims(c_img, 0)/255.0

    ax1.imshow(c_img[0])

    ax1.set_title('Image: ' + c_img_name)

    ax2.imshow(masks_as_color(sub.query('ImageId=="{}"'.format(c_img_name))['EncodedPixels']))

    ax2.set_title('Prediction')
sub1 = pd.read_csv('../input/airbus-ship-detection/sample_submission_v2.csv')

sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])

sub1['EncodedPixels'] = None

print(len(sub1), len(sub))



sub = pd.concat([sub, sub1])

print(len(sub))

sub.to_csv('submission.csv', index=False)

sub.head()