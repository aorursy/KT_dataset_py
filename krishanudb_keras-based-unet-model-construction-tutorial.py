import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))
ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
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

def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
masks = pd.read_csv(os.path.join('../input/', 'train_ship_segmentations.csv'))
print(masks.shape[0], ' masks found')
print(masks['ImageId'].value_counts().shape[0], ' images found')
masks.head()
# First need to import some libararies
import skimage
from skimage.io import imread

train_images = os.listdir(train_image_dir)
train_temp = np.random.choice(train_images, 2000) # We choose 2000 random images every time

train_temp_img = []
train_temp_mask = []

for img in train_temp:
    train_temp_img.append(imread(train_image_dir + '/' + img).astype('uint8'))
    train_temp_mask.append(masks_as_image(masks.query('ImageId=="'+img+'"')['EncodedPixels']))
    
train_temp_img = np.array(train_temp_img)
train_temp_mask = np.array(train_temp_mask)

print(train_temp_img.shape, train_temp_mask.shape)
    
random = np.random.choice(2000, 1)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].imshow(train_temp_img[random[0]])
axes[1].imshow(train_temp_mask[random[0]][:, :, 0])
plt.show()
fg, axes = plt.subplots(5, 2, figsize=(15, 50))

randoms = np.random.choice(range(2000), 5)

axes[0, 0].imshow(train_temp_img[randoms[0]])
axes[0, 1].imshow(train_temp_mask[randoms[0]][:, :, 0])
axes[1, 0].imshow(train_temp_img[randoms[1]])
axes[1, 1].imshow(train_temp_mask[randoms[1]][:, :, 0])
axes[2, 0].imshow(train_temp_img[randoms[2]])
axes[2, 1].imshow(train_temp_mask[randoms[2]][:, :, 0])
axes[3, 0].imshow(train_temp_img[randoms[3]])
axes[3, 1].imshow(train_temp_mask[randoms[3]][:, :, 0])
axes[4, 0].imshow(train_temp_img[randoms[4]])
axes[4, 1].imshow(train_temp_mask[randoms[4]][:, :, 0])

plt.show()
        
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    loss = (1 - jac) * smooth
#     print(loss.shape)
#     print(loss)
    return loss

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

from keras import regularizers
from keras.layers import BatchNormalization as BatchNorm
from keras.models import Model, load_model
from keras.layers import Input, Reshape, Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import layers
def downsampling(x, level, filters, kernel_size, num_convs, conv_strides=1, activation = 'relu', batch_norm = False, pool_size = 2, pool_strides = 2, regularizer = None, regularizer_param = 0.001):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=conv_strides, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = 'downsampling_' + str(level) + '_conv_' + str(i))(x)
        if batch_norm:
            x = BatchNorm(name = 'downsampling_' + str(level) + '_batchnorm_' + str(i))(x)
        x = Activation(activation, name = 'downsampling_' + str(level) + '_activation_' + str(i))(x)
    skip = x
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    return x, skip

def bottleneck_dilated(x, filters, kernel_size, num_convs = 6, activation = 'relu', batch_norm = False, last_activation = False, regularizer = None, regularizer_param = 0.001):
#     assert num_convs == len(conv_strides)
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    skips = []
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate = 2 ** i, activation='relu', padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = 'bottleneck_skip_' + str(i))(x)
        skips.append(x)
    x = layers.add(skips)
    if last_activation:
        x = Activation('relu')(x)
    return x
    
def bottleneck(x, filters, kernel_size, num_convs, conv_strides=1, activation = 'relu', batch_norm = False, pool_size = 2, pool_strides = 2, regularizer = None, regularizer_param = 0.001):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = 'bottleneck_' + str(i))(x)
        if batch_norm:
            x = BatchNorm()(x)
        x = Activation(activation)(x)
    return x

def upsampling(x, level, skip, filters, kernel_size, num_convs, conv_strides=1, activation = 'relu', batch_norm = False, conv_transpose = True, upsampling_size = 2, upsampling_strides = 2, regularizer = None, regularizer_param = 0.001):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    if conv_transpose:
        x = Conv2DTranspose(filters=filters, kernel_size = upsampling_size, strides=upsampling_strides, name = 'upsampling_' + str(level) + '_conv_trans_' + str(level))(x)
    else:
        x = UpSampling2D((upsampling_size), name = 'upsampling_' + str(level) + '_ups_' + str(i))(x)
    x = Concatenate()([x, skip])
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = 'upsampling_' + str(level) + '_conv_' + str(i))(x)
        if batch_norm:
            x = BatchNorm(name = 'upsampling_' + str(level) + '_batchnorm_' + str(i))(x)
        x = Activation(activation, name = 'upsampling_' + str(level) + '_activation_' + str(i))(x)
    return x

def model_simple_unet_initializer(num_classes, num_levels, num_layers = 2, num_bottleneck = 2, filter_size_start = 16, batch_norm = False, kernel_size = 3, bottleneck_dilation = False, bottleneck_sum_activation = False, regularizer = None, regularizer_param = 0.001):
    inputs = Input((img_shape))
    x = inputs
    skips = []
    for i in range(num_levels):
        x, skip = downsampling(x, i, filter_size_start * (2 ** i), kernel_size, num_layers, batch_norm=True, regularizer= regularizer, regularizer_param=regularizer_param)
        skips.append(skip)
    if bottleneck_dilation:
        x = bottleneck_dilated(x, filter_size_start * (2 ** num_levels), kernel_size, num_bottleneck, batch_norm=True, last_activation=bottleneck_sum_activation, regularizer= regularizer, regularizer_param=regularizer_param)
    else:
        x = bottleneck(x, filter_size_start * (2 ** num_levels), kernel_size, num_bottleneck, batch_norm=True, regularizer=regularizer, regularizer_param=regularizer_param)
    for j in range(num_levels):
        x = upsampling(x, j, skips[num_levels - j - 1], filter_size_start * (2 ** (num_levels - j - 1)), kernel_size, num_layers, batch_norm=True, regularizer= regularizer, regularizer_param=regularizer_param)
    outputs = Conv2D(filters=num_classes, kernel_size=1, strides=1, padding='same', activation='softmax', name = 'output_softmax')(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer='adam', loss=jaccard_distance_loss, metrics=[dice_coef])
    model.summary()
    return model

def model_train(model, x, y, epochs, num_test, early_stopper, patience_lr, model_name):
    num_data = x.shape[0]
    num_train = num_data - num_test
    early_stopper = EarlyStopping(patience=early_stopper, verbose=1)
    reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor = 0.75, patience = patience_lr, verbose=1)
    checkpointer = ModelCheckpoint(model_name + '.h5', verbose=1, save_best_only=True)
    checkpointer_train = ModelCheckpoint(model_name + 'best_train.h5', monitor='loss', verbose=1, save_best_only=True)
    results = model.fit(x[0:num_train], y[0:num_train], validation_data=(x[num_train:], y[num_train:]), batch_size=5, epochs=epochs, callbacks=[early_stopper, checkpointer, checkpointer_train, reduce_learning_rate])
    return model, results
img_height = 768
img_width = 768
num_channels = 3
img_shape = (img_height, img_width, num_channels)
num_classes = 1
model1 = model_simple_unet_initializer(1, 4, 2, 5, 16, True, 3, True, False, 'l2', 0.001)
model1, results1 = model_train(model1, train_temp_img, train_temp_mask, 30, 50, 8, 6, 'model1')
