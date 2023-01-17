# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import math
import random
import gc; gc.enable() # memory is tight
import matplotlib.pyplot as plt
%matplotlib inline
from skimage.data import imread
from skimage.morphology import label
from pathlib import Path
from math import ceil
from functools import partial
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras import Model
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.layers import LeakyReLU, Add, ZeroPadding2D, Conv2DTranspose
from keras.layers import Conv2D, Concatenate, concatenate, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(os.listdir(".."))
INPUT_PATH = "../input"
BATCH_SIZE = 8
IMG_SCALING = (2, 2)
EDGE_CROP = 16
AUGMENT_BRIGHTNESS = False
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'DECONV'
MAX_TRAIN_STEPS = 200
NET_SCALING = None
DATA_PATH = INPUT_PATH
TRAIN = os.path.join(DATA_PATH, "train_v2")
MASKS = os.path.join(DATA_PATH, "train_ship_segmentations_v2.csv")
TEST = os.path.join(DATA_PATH, "test_v2")
TEST_MASKS = os.path.join(DATA_PATH, "sample_submission_v2.csv")
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        data_path = TRAIN
    elif "Test" in image_type:
        data_path = TEST
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}".format(image_id))

def get_image_data(image_id, image_type, **kwargs):
    img = _get_image_data_opencv(image_id, image_type, **kwargs)
    img = img.astype('uint8')
    return img

def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
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

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
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

# https://github.com/ternaus/TernausNet/blob/master/Example.ipynb
def mask_overlay(image, mask):
    """
    Helper function to visualize mask
    """
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.75, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
masks = pd.read_csv(MASKS)
file_names = os.listdir(TRAIN)
print("number of records in train_ship_segmentations_v2: {}".format(len(masks)))
print("Train files :",len(file_names))
print("number of train images: {}".format(len(masks.ImageId.unique())))
masks.head()
masks.info()
sample = masks[~masks.EncodedPixels.isna()].sample(25)

fig, ax = plt.subplots(5, 5, sharex='col', sharey='row')
fig.set_size_inches(25, 25)

for i, imgid in enumerate(sample.ImageId):
    col = i % 5
    row = i // 5
    
    img = get_image_data(imgid, "Train")
    
    ax[row, col].set_title(imgid)
    ax[row, col].imshow(img)
# Show imgs with size < 50kB and with ships
# sample = masks[masks.ImageId.apply(lambda x: (os.stat(get_filename(x, "Train")).st_size/1024) < 50) & ~masks.EncodedPixels.isna()].sample(25)

sample = masks[masks.EncodedPixels.isna()].sample(25)

fig, ax = plt.subplots(5, 5, sharex='col', sharey='row')
fig.set_size_inches(25, 25)

for i, imgid in enumerate(sample.ImageId):
    col = i % 5
    row = i // 5
    
    img = get_image_data(imgid, "Train")
    
    ax[row, col].set_title(imgid)
    ax[row, col].imshow(img)
NUM_IMG = 20
NUM_COL = 2
NUM_MASKS = 2
IMG_SIZE = 25

sample = masks[~masks.EncodedPixels.isna()].sample(NUM_IMG)

# Show imgs with size < 50kB and with ships
# sample = masks[masks.ImageId.apply(lambda x: (os.stat(get_filename(x, "Train")).st_size/1024) < 42) & ~masks.EncodedPixels.isna()].sample(NUM_IMG)
# sample = masks[masks.ImageId.apply(lambda x: (os.stat(get_filename(x, "Train")).st_size/1024) < 40)].sample(NUM_IMG)

number_of_rows = ceil(NUM_IMG / NUM_COL)
number_of_cols = NUM_COL * NUM_MASKS

fig, ax = plt.subplots(number_of_rows, number_of_cols, sharex='col', sharey='row')
fig.set_size_inches(IMG_SIZE, IMG_SIZE * (number_of_rows / number_of_cols))

for i, imgid in enumerate(sample.ImageId):
    col = (i % NUM_COL) * 2
    row = i // NUM_COL
    
    img = get_image_data(imgid, "Train")
    
    # if the ship is in the image, show next to original image, image with mask 
    if all(isinstance(x, str) for x in masks[masks.ImageId == imgid].EncodedPixels):
        decoded_masks = masks[masks.ImageId == imgid].EncodedPixels.apply(lambda x: rle_decode(x))
        mask = sum(decoded_masks)
        mask = np.expand_dims(mask,axis=2)
        mask = np.repeat(mask,3,axis=2).astype('uint8')*255

        img_masked = mask_overlay(img, mask)
    else:
        img_masked = np.full((img.shape[0],img.shape[1],3), 255)
        
    ax[row, col].set_title(imgid)
    ax[row, col].imshow(img)
    ax[row, col+1].imshow(img_masked)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 7))
rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
img_0 = masks_as_image(rle_0)
ax1.imshow(img_0[:, :, 0])
ax1.set_title('Image$_0$')
rle_1 = multi_rle_encode(img_0)
img_1 = masks_as_image(rle_1)
ax2.imshow(img_1[:, :, 0])
ax2.set_title('Image$_1$')
print('Check Decoding->Encoding',
      'RLE_0:', len(rle_0), '->',
      'RLE_1:', len(rle_1))
ships = masks[~masks.EncodedPixels.isna()].ImageId.unique()
noships = masks[masks.EncodedPixels.isna()].ImageId.unique()

plt.bar(['Ships', 'No Ships'], [len(ships), len(noships)]);
plt.ylabel('Number of Images');
# groupby ImageId and make list from EncodedPixels
unique_img_ids = masks.groupby('ImageId', as_index=False)['EncodedPixels'].agg({'EncodedPixels':(lambda x: list(x))})
# count of ships in 1 img
unique_img_ids['ships_count'] = unique_img_ids['EncodedPixels'].map(lambda x: len(x) if isinstance(x[0], str) else 0)

# some files are too small/corrupt
unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda img_id: os.stat(get_filename(img_id, "Train")).st_size/1024)
unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50] # keep only 50kb files
# unique_img_ids['file_size_kb'].hist()

unique_img_ids[unique_img_ids.ships_count > 0]['ships_count'].hist(bins=np.arange(12))
unique_img_ids.head()
train_df, valid_df = train_test_split(unique_img_ids, 
                 test_size = 0.05, 
                 stratify = unique_img_ids['ships_count'])
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')
def sample_ships(in_df, base_rep_val=30000):
    if in_df['ships_count'].values[0]==0:
        return in_df.sample(base_rep_val) # undersample img without ships
    else:
        return in_df
    
balanced_train_df = train_df.groupby('ships_count').apply(sample_ships)
balanced_train_df['ships_count'].hist(bins=np.arange(12))
sample_ships_valid = partial(sample_ships, base_rep_val = 1500)
balanced_valid_df = valid_df.groupby('ships_count').apply(sample_ships_valid)
balanced_valid_df['ships_count'].hist(bins=np.arange(12))
def make_image_gen(in_df, batch_size = BATCH_SIZE):
    out_rgb = []
    out_mask = []
    while True:
        in_df = in_df.sample(frac = 1)    # schuffle
        for index, row in in_df.iterrows():
            c_img = get_image_data(row['ImageId'], "Train")
            c_mask = masks_as_image(row['EncodedPixels'])
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
train_gen = make_image_gen(balanced_train_df)
train_x, train_y = next(train_gen)

print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())

SHOW_MAX_IMG = 4
IMG_SIZE = 6
num_of_row = min(SHOW_MAX_IMG//2, train_x.shape[0]//2)
fig, ax = plt.subplots(num_of_row, 4, sharex='col', sharey='row', figsize = (IMG_SIZE*4, IMG_SIZE*num_of_row))
for i, imgid in enumerate(train_x):
    if i >= num_of_row*2:
        break
    col = i // num_of_row * 2
    row = i % num_of_row
    ax[row, col].imshow(train_x[i])
    ax[row, col+1].imshow(train_y[i][:, :, 0])
VALID_IMG_COUNT = 400        #beware on RAM usage!  
if IMG_SCALING == (2, 2):
    VALID_IMG_COUNT = 800
if IMG_SCALING == (3, 3):
    VALID_IMG_COUNT = 1200
valid_x, valid_y = next(make_image_gen(balanced_valid_df, VALID_IMG_COUNT))
print('x', valid_x.shape, train_x.min(), train_x.max())
print('y', valid_y.shape, train_y.min(), train_y.max())
# treshold mask_img values to 0.0 or 1.0
def treshold_mask(mask_img, threshold = 0.5):
    ret, thresh = cv2.threshold(mask_img, 0.5, 1.0, cv2.THRESH_BINARY)
    # returns back the third dimension
    return np.reshape(thresh, (thresh.shape[0], thresh.shape[1], -1))

dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
#                   rotation_range = 15, 
#                   width_shift_range = 0.1, 
#                   height_shift_range = 0.1, 
#                   shear_range = 0.01,
#                   zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')
# brightness can be problematic since it seems to change the labels differently from the images 
if AUGMENT_BRIGHTNESS:
    dg_args['brightness_range'] = [0.5, 1.5]
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
        
        # treshold g_y values to 0.0 or 1.0
        yield next(g_x)/255.0, np.asarray([treshold_mask(x) for x in next(g_y)])
cur_gen = create_aug_gen(train_gen, seed = 42)
t_x, t_y = next(cur_gen)

 
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())

SHOW_MAX_IMG = 4
IMG_SIZE = 6
num_of_row = min(SHOW_MAX_IMG//2, train_x.shape[0]//2)
fig, ax = plt.subplots(num_of_row, 4, sharex='col', sharey='row', figsize = (IMG_SIZE*4, IMG_SIZE*num_of_row))
for i, imgid in enumerate(t_x):
    if i >= num_of_row*2:
        break
    col = i // num_of_row * 2
    row = i % num_of_row
    ax[row, col].imshow(t_x[i])
    ax[row, col+1].imshow(t_y[i][:, :, 0])
test_gen = make_image_gen(balanced_train_df, batch_size = 4)
# np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
in_x, in_y = next(test_gen)
seed = np.random.choice(range(9999))
g_x = image_gen.flow(255*in_x,
                     batch_size = t_x.shape[0], 
                     seed = seed, 
                     shuffle=True)
g_y = label_gen.flow(in_y, 
                     batch_size = in_x.shape[0], 
                     seed = seed, 
                     shuffle=True)
t_x = next(g_x)
t_x /= 255
t_y = next(g_y)
t_y = np.asarray([treshold_mask(x) for x in t_y])

fig, ax = plt.subplots(4, 4, sharex='col', sharey='row', figsize = (24, 24))
for i, imgid in enumerate(t_x):
    ax[i, 0].imshow(in_x[i])
    ax[i, 1].imshow(in_y[i][:, :, 0])
    ax[i, 2].imshow(t_x[i])
    ax[i, 3].imshow(t_y[i][:, :, 0])
gc.collect()
def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x
def UXception(input_shape=(None, None, 3)):

    backbone = Xception(input_shape=input_shape,weights='imagenet',include_top=False)
    input = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[121].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.1)(pool4)
    
     # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)
    
    # 10 -> 20
    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.1)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    
    # 10 -> 20
    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(0.1)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)
    
    # 20 -> 40
    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)
    
    # 40 -> 80
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)
    
    # 80 -> 160
    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)   
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    
    uconv0 = Dropout(0.1/2)(uconv0)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv0)    
    
    model = Model(input, output_layer)
    model.name = 'u-xception'

    return model
K.clear_session()
seg_model = UXception(input_shape=(t_x.shape[1],t_x.shape[1],3))
# seg_model.summary()
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred, smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return K.mean(binary_crossentropy(y_true, y_pred)) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return K.mean(binary_crossentropy(y_true, y_pred)) - K.log(1. - dice_loss(y_true, y_pred))

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)
weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef", 
                      mode="max", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=bce_logdice_loss, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
MAX_TRAIN_STEPS = 500
NB_EPOCHS = 16
step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0]//BATCH_SIZE)
train_gen = create_aug_gen(make_image_gen(balanced_train_df))
# train_gen = make_image_gen(balanced_train_df)
loss_history = [seg_model.fit_generator(train_gen, 
                             steps_per_epoch=step_count, 
                             epochs=NB_EPOCHS, 
                             validation_data=(valid_x, valid_y),
                             callbacks=callbacks_list,
                            workers=1 # the generator is not very thread safe
                                       )]
def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['true_positive_rate'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_true_positive_rate'] for mh in loss_history]),
                     'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')
     
    _ = ax3.plot(epich, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                     'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')
    
    _ = ax4.plot(epich, np.concatenate(
        [mh.history['dice_coef'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_dice_coef'] for mh in loss_history]),
                     'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')

show_loss(loss_history)
seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')
pred_y = seg_model.predict(valid_x)
print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())
fig, ax = plt.subplots(1, 1, figsize = (8, 6))
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
test_paths = os.listdir(TEST)
print(len(test_paths), 'test images found')
NUMBER_OF_IMG = 40
IMG_SIZE = 6
num_of_row = math.ceil(NUMBER_OF_IMG / 2)

fig, ax = plt.subplots(num_of_row, 4, sharex='col', sharey='row', figsize = (IMG_SIZE*4, IMG_SIZE*num_of_row))

random.shuffle(test_paths)
for i, imgid in enumerate(test_paths[:NUMBER_OF_IMG]):
    col = i // num_of_row * 2
    row = i % num_of_row
    
    img = get_image_data(imgid, "Test")
    img = np.expand_dims(img, 0)/255.0
    img_seg = fullres_model.predict(img)
    
    ax[row, col].imshow(img[0])
    ax[row, col+1].imshow(img_seg[0][:, :, 0], vmin = 0, vmax = 1)
fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
fig.set_size_inches(24, 7)

mask = masks.EncodedPixels.isna()
for i, (msk, label) in enumerate(zip([mask, ~mask], ['No Ships', 'Ships'])):
    _ids = masks[msk].ImageId.sample(250)
    imgs = np.array([get_image_data(_id, "Train") for _id in _ids])
    
    red = imgs[:, :, :, 0]
    green = imgs[:, :, :, 1]
    blue = imgs[:, :, :, 2]
    
    ax[i].plot(np.bincount(red.ravel()), color='orangered', label='red', lw=2)
    ax[i].plot(np.bincount(green.ravel()), color='yellowgreen', label='green', lw=2)
    ax[i].plot(np.bincount(blue.ravel()), color='skyblue', label='blue', lw=2)
    ax[i].legend()
    ax[i].title.set_text(label)
def apply_masks_to_img(img, _id, df):
    '''Apply masks to image given img, its id and the dataframe.'''
    masks = df[df.ImageId == _id].EncodedPixels.apply(lambda x: rle_decode(x)).tolist()
    masks = sum(masks)
    return img * masks.reshape(img.shape[0], img.shape[1], 1)


fig, ax = plt.subplots(1, 2, sharex='col')#, sharey='row')
fig.set_size_inches(24, 7)

mask = masks.EncodedPixels.isna()
for i, (msk, label) in enumerate(zip([mask, ~mask], ['No Ships', 'Ships'])):
    _ids = masks[msk].ImageId.sample(250)
    imgs = [get_image_data(_id, "Train") for _id in _ids]
    
    # if we have an encoding to decode
    if i == 1:
        imgs = [apply_masks_to_img(i, _id, masks) for (i, _id) in zip(imgs, _ids)]

    imgs = np.array(imgs)
    red = imgs[:, :, :, 0]
    green = imgs[:, :, :, 1]
    blue = imgs[:, :, :, 2]
    
    # skip bincount index 0 to avoid the masked pixels to overpower the others.
    ax[i].plot(np.bincount(red.ravel())[1:], color='orangered', label='red', lw=2)
    ax[i].plot(np.bincount(green.ravel())[1:], color='yellowgreen', label='green', lw=2)
    ax[i].plot(np.bincount(blue.ravel())[1:], color='skyblue', label='blue', lw=2)
    ax[i].legend()
    ax[i].title.set_text(label)