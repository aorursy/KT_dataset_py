BATCH_SIZE = 48

EDGE_CROP = 16

GAUSSIAN_NOISE = 0.1

UPSAMPLE_MODE = 'SIMPLE'

# downsampling inside the network

NET_SCALING = (1, 1)

# downsampling in preprocessing

IMG_SCALING = (3, 3)

# number of validation images to use

VALID_IMG_COUNT = 900

# maximum number of steps_per_epoch in training

MAX_TRAIN_STEPS = 9

MAX_TRAIN_EPOCHS = 99

AUGMENT_BRIGHTNESS = False
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from skimage.io import imread

from skimage.morphology import binary_opening, disk, label
BASE_DIR = '/kaggle/input/airbus-ship-detection/'

TRAIN_DIR = BASE_DIR + '/train_v2/'

TEST_DIR = BASE_DIR + '/test_v2/'
train = os.listdir(TRAIN_DIR)

test = os.listdir(TEST_DIR)



print(f"Train files: {len(train)}. ---> {train[:3]}")

print(f"Test files :  {len(test)}. ---> {test[:3]}")
from PIL import Image



Image.open(TRAIN_DIR+train[0])
masks = pd.read_csv(os.path.join(BASE_DIR, 'train_ship_segmentations_v2.csv'))

not_empty = pd.notna(masks.EncodedPixels)

print(not_empty.sum(), 'masks in', masks[not_empty].ImageId.nunique(), 'images')

print((~not_empty).sum(), 'empty images in', masks.ImageId.nunique(), 'total images')

masks.head()
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



def showImage(image_name):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (16, 5))

    rle_0 = masks.query('ImageId=="'+image_name+'"')['EncodedPixels']

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

masks.drop(['ships'], axis=1, inplace=True)

print(unique_img_ids.loc[unique_img_ids.ships>=2].head())

showImage(unique_img_ids.loc[unique_img_ids.ships>=2].iloc[0].ImageId)
unique_img_ids['ships'].hist(bins=unique_img_ids['ships'].max())

print('Max of ships : ',unique_img_ids['ships'].max())

print('Avg of ships : ',unique_img_ids['ships'].mean())
SAMPLES_PER_GROUP = 4000

balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)

print(balanced_train_df.shape[0], 'masks')
from sklearn.model_selection import train_test_split

train_ids, valid_ids = train_test_split(balanced_train_df, 

                 test_size = 0.2, 

                 stratify = balanced_train_df['ships'])

train_df = pd.merge(masks, train_ids)

valid_df = pd.merge(masks, valid_ids)

print(train_df.shape[0], 'training masks')

print(valid_df.shape[0], 'validation masks')
BATCH_SIZE = 100

def make_image_gen(in_df, batch_size = BATCH_SIZE):

    all_batches = list(in_df.groupby('ImageId'))

    out_rgb = []

    out_mask = []

    while True:

        np.random.shuffle(all_batches)

        for c_img_id, c_masks in all_batches:

            rgb_path = os.path.join(TRAIN_DIR, c_img_id)

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
import gc; gc.enable() 



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

    

input_img = layers.Input(t_x.shape[1:], name = 'RGB_Input')

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
import keras.backend as K

from keras.optimizers import Adam

from keras.losses import binary_crossentropy



## intersection over union

def IoU(y_true, y_pred, eps=1e-6):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])

    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection

    return -K.mean( (intersection + eps) / (union + eps), axis=0)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('seg_model')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,

                                   patience=1, verbose=1, mode='min',

                                   min_delta=0.0001, cooldown=0, min_lr=1e-8)



early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,

                      patience=20) # probably needs to be more patient, but kaggle time is limited



callbacks_list = [checkpoint, early, reduceLROnPlat]
def fit():

    seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])

    

    step_count = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)

    aug_gen = create_aug_gen(make_image_gen(train_df))

    loss_history = [seg_model.fit_generator(aug_gen,

                                 steps_per_epoch=step_count,

                                 epochs=MAX_TRAIN_EPOCHS,

                                 validation_data=(valid_x, valid_y),

                                 callbacks=callbacks_list,

                                workers=1 # the generator is not very thread safe

                                           )]

    return loss_history



while True:

    loss_history = fit()

    if np.min([mh.history['val_loss'] for mh in loss_history]) < -0.2:

        break
if IMG_SCALING is not None:

    fullres_model = models.Sequential()

    fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))

    fullres_model.add(seg_model)

    fullres_model.add(layers.UpSampling2D(IMG_SCALING))

else:

    fullres_model = seg_model

fullres_model.save('fullres_model.h5')
def raw_prediction(img, path=TEST_DIR):

    img = imread(os.path.join(path, img))

    img = np.expand_dims(img, 0)/255

    seg = fullres_model.predict(img)[0]

    return seg, img[0]
from skimage.morphology import binary_opening, disk



def smooth(seg):

    return binary_opening(seg>0.99, np.expand_dims(disk(2), -1))
def predict(img, path=TEST_DIR):

    seg, img = raw_prediction(img, path=path)

    return smooth(seg), img
test_paths = np.array(os.listdir(TEST_DIR))

print('{} test images found'.format(len(test_paths)))
from tqdm import tqdm_notebook



def pred_encode(img, **kwargs):

    cur_seg, _ = predict(img)

    cur_rles = multi_rle_encode(cur_seg, **kwargs)

    return [[img, rle] for rle in cur_rles if rle is not None]



out_pred_rows = []

for c_img_name in tqdm_notebook(test_paths[:30000]): ## only a subset as it takes too long to run

    out_pred_rows += pred_encode(c_img_name, min_max_threshold=1.0)
sub = pd.DataFrame(out_pred_rows)

sub.columns = ['ImageId', 'EncodedPixels']

sub = sub[sub.EncodedPixels.notnull()]

sub.head()
TOP_PREDICTIONS=5

fig, m_axs = plt.subplots(TOP_PREDICTIONS, 2, figsize = (9, TOP_PREDICTIONS*5))

[c_ax.axis('off') for c_ax in m_axs.flatten()]



for (ax1, ax2), c_img_name in zip(m_axs, sub.ImageId.unique()[:TOP_PREDICTIONS]):

    c_img = imread(os.path.join(TEST_DIR, c_img_name))

    c_img = np.expand_dims(c_img, 0)/255.0

    ax1.imshow(c_img[0])

    ax1.set_title('Image: ' + c_img_name)

    ax2.imshow(masks_as_color(sub.query('ImageId=="{}"'.format(c_img_name))['EncodedPixels']))

    ax2.set_title('Prediction')
sub1 = pd.read_csv(BASE_DIR+'sample_submission_v2.csv')

sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])

sub1['EncodedPixels'] = None

print(len(sub1), len(sub))



sub = pd.concat([sub, sub1])

print(len(sub))

sub.to_csv('submission.csv', index=False)

sub.head()