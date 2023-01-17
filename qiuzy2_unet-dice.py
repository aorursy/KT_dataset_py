# import re

# import tensorflow as tf

# import numpy as np

# from matplotlib import pyplot as plt

# print("Tensorflow version " + tf.__version__)

# AUTO = tf.data.experimental.AUTOTUNE

# from kaggle_datasets import KaggleDatasets

# # Detect hardware, return appropriate distribution strategy

# try:

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

#     print('Running on TPU ', tpu.master())

# except ValueError:

#     tpu = None



# if tpu:

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# else:

#     strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



# print("REPLICAS: ", strategy.num_replicas_in_sync)



BATCH_SIZE = 50

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

ship_dir = '../input'

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
ship_dir1 = '../input'

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
unique_img_ids['ships'].hist(bins=unique_img_ids['ships'].max())
unique_img_ids = unique_img_ids[unique_img_ids['ships']!=0]
SAMPLES_PER_GROUP = 1800

unique_img_ids['grouped_ship_count'] = unique_img_ids['ships'].map(lambda x: (x+2)//3)

balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)

print(balanced_train_df.shape[0], 'masks')
balanced_train_df=balanced_train_df.reset_index(drop = True)#删除原来的索引。

balanced_train_df=balanced_train_df.sample(frac=1.0)
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
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))

# batch_rgb = montage_rgb(train_x)

# batch_seg = montage(train_y[:, :, :, 0])

# ax1.imshow(batch_rgb)

# ax1.set_title('Images')

# ax2.imshow(batch_seg)

# ax2.set_title('Segmentations')

# ax3.imshow(mark_boundaries(batch_rgb, 

#                            batch_seg.astype(int)))

# ax3.set_title('Outlined Ships')

# fig.savefig('overview.png')
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
# cur_gen = create_aug_gen(train_gen)

# t_x, t_y = next(cur_gen)

# print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())

# print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())

# # only keep first 9 samples to examine in detail

# t_x = t_x[:9]

# t_y = t_y[:9]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))

# ax1.imshow(montage_rgb(t_x), cmap='gray')

# ax1.set_title('images')

# ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray_r')

# ax2.set_title('ships')
gc.collect()
from keras import models, layers

# Build U-Net model

def upsample_conv(filters, kernel_size, strides, padding):

    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

def upsample_simple(filters, kernel_size, strides, padding):

    return layers.UpSampling2D(strides)



if UPSAMPLE_MODE=='DECONV':

    upsample=upsample_conv

else:

    upsample=upsample_simple

    

input_img = layers.Input([256,256,3], name = 'RGB_Input')

pp_in_layer = input_img



if NET_SCALING is not None:

    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

    

pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)

pp_in_layer = layers.BatchNormalization()(pp_in_layer)



c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (pp_in_layer)

c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)

p1 = layers.MaxPooling2D((2, 2)) (c1)



c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)

c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)

p2 = layers.MaxPooling2D((2, 2)) (c2)



c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)

c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)

p3 = layers.MaxPooling2D((2, 2)) (c3)



c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)

c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)

p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)





c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p4)

c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c5)



u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = layers.concatenate([u6, c4])

c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)

c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)



u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = layers.concatenate([u7, c3])

c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)

c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)



u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = layers.concatenate([u8, c2])

c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)

c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)



u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = layers.concatenate([u9, c1], axis=3)

c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)

c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)



d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)

# d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)

# d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)

if NET_SCALING is not None:

    d = layers.UpSampling2D(NET_SCALING)(d)



seg_model = models.Model(inputs=[input_img], outputs=[d])

seg_model.summary()
from numpy import array



def dice_loss(input, target):

    input = K.sigmoid(input)

    smooth = 1.0



#     input=array(input,'f')

#     target=array(target,'f')

    

    iflat = K.flatten(input)

    tflat = K.flatten(target)

    intersection = K.sum((iflat * tflat))

    

    return ((2.0 * intersection + smooth) / (K.sum(iflat)+ K.sum(tflat) + smooth))
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



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

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

    

    max_val = K.clip(-input, 0, 100000)

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



# with strategy.scope():

seg_model.compile(optimizer=Adam(1e-2, decay=1e-6), loss=dice_loss, metrics=[IoU, zero_IoU, 'binary_accuracy'])
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
BATCH_SIZE = 50

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

    epochs = np.concatenate([mh.epoch for mh in loss_history])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    

    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',

                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')

    ax1.legend(['Training', 'Validation'])

    ax1.set_title('Loss')

    

    _ = ax2.plot(epochs, np.concatenate([mh.history['IoU'] for mh in loss_history]), 'b-',

                 epochs, np.concatenate([mh.history['val_IoU'] for mh in loss_history]), 'r-')

    ax2.legend(['Training', 'Validation'])

    ax2.set_title('Binary Accuracy (%)')



show_loss(loss_history)
seg_model.load_weights(weight_path)

seg_model.save('seg_model.h5')
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
sub1 = pd.read_csv('../input/sample_submission_v2.csv')

sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])

sub1['EncodedPixels'] = None

print(len(sub1), len(sub))



sub = pd.concat([sub, sub1])

print(len(sub))

sub.to_csv('submission.csv', index=False)

sub.head()