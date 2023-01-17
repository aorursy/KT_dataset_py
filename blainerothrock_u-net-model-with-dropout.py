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
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread

import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap

from skimage.segmentation import mark_boundaries

from skimage.util import montage2d as montage

from skimage.morphology import binary_opening, disk, label

import gc; gc.enable() # memory is tight



montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

ship_dir = '../input'

train_image_dir = os.path.join(ship_dir, 'train_v2')

test_image_dir = os.path.join(ship_dir, 'test_v2')



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
masks = pd.read_csv(os.path.join('../input/', 'train_ship_segmentations_v2.csv'))

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
SAMPLES_PER_GROUP = 2000

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
from keras import models, Input

from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, UpSampling2D, GaussianNoise, BatchNormalization, concatenate, Cropping2D, ZeroPadding2D, AvgPool2D



class UNetParams():

    def __init__(self):

        self.batch_size = 4

        self.edge_crop = 16

        self.num_epochs= 10

        self.gaussian_noise = 0.1

        self.img_scaling = (3, 3)

        self.max_train_steps = 5000

        self.validation_set_size = 1000

        self.augment_brightness = False

        

unet_params = UNetParams()



def conv_down(filter_, in_layer, name, kernel=(3, 3), activation='relu', padding='same'):

    l = Conv2D(filter_, kernel, activation=activation, padding=padding, name=name+'_conv1')(in_layer)

    l = Conv2D(filter_, kernel, activation=activation, padding=padding, name=name+'_conv2')(l)

    return l



def pool(in_layer, name, pool_size=(2, 2)):

    return MaxPooling2D(pool_size, name=name+'_pool')(in_layer)



def conv_up(filter_, in_layer, conv_down_layer, name, upsample_size=(2, 2), kernel=(3, 3), activation='relu', padding='same'):

    l = UpSampling2D(upsample_size, name=name+'_upsample')(in_layer)

    l = concatenate([l, conv_down_layer], name=name+'_concat')

    l = Dropout(0.2)(l)

    l = Conv2D(filter_, kernel, activation=activation, padding=padding, name=name+'_conv1')(l)

    l = Conv2D(filter_, kernel, activation=activation, padding=padding, name=name+'_conv2')(l)

    return l



input_img = Input(t_x.shape[1:], name = 'RGB_Input')

input_layer = GaussianNoise(unet_params.gaussian_noise)(input_img)

input_layer = BatchNormalization()(input_layer)



d1 = conv_down(8, input_img, name='d1')

dp1 = pool(d1, name='d1')

dp1 = Dropout(0.2)(dp1)

d2 = conv_down(16, dp1, name='d2')

dp2 = pool(d2, name='d2')

dp2 = Dropout(0.2)(dp2)

d3 = conv_down(32, dp2, name='d3')

dp3 = pool(d3, name='d3')

dp3 = Dropout(0.2)(dp3)

d4 = conv_down(64, dp3, name='d4')

dp4 = pool(d4, name='d4')

dp4 = Dropout(0.2)(dp4)

b = conv_down(128, dp4, name='b')

u1 = conv_up(64, b, d4, name='u1')

u2 = conv_up(32, u1, d3, name='u2')

u3 = conv_up(16, u2, d2, name='u3')

u4 = conv_up(8, u3, d1, name='u4')



out = Conv2D(1, (1, 1), activation='sigmoid', name='out_conv1')(u4)

out = Cropping2D((unet_params.edge_crop, unet_params.edge_crop), name='out_crop')(out)

out = ZeroPadding2D((unet_params.edge_crop, unet_params.edge_crop), name='out_pad')(out)



seg_model = models.Model(inputs=[input_img], outputs=[out], name="UNet")

seg_model.summary()
import keras.backend as K

from keras.optimizers import Adam

from keras.losses import binary_crossentropy



## intersection over union

def IoU(y_true, y_pred, eps=1e-6):

    if np.max(y_true) == 0.0:

        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros

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



loss_history = fit()
def show_loss(loss_history):

    epochs = np.concatenate([mh.epoch for mh in loss_history])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    

    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',

                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')

    ax1.legend(['Training', 'Validation'])

    ax1.set_title('Loss')

    

    _ = ax2.plot(epochs, np.concatenate([mh.history['binary_accuracy'] for mh in loss_history]), 'b-',

                 epochs, np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history]), 'r-')

    ax2.legend(['Training', 'Validation'])

    ax2.set_title('Binary Accuracy (%)')



show_loss(loss_history)
seg_model.load_weights(weight_path)

seg_model.save('seg_model.h5')
pred_y = seg_model.predict(valid_x)

print(pred_y.shape, pred_y.min(axis=0).max(), pred_y.max(axis=0).min(), pred_y.mean())
fig, ax = plt.subplots(1, 1, figsize = (6, 6))

ax.hist(pred_y.ravel(), np.linspace(0, 1, 20))

ax.set_xlim(0, 1)

ax.set_yscale('log', nonposy='clip')
if IMG_SCALING is not None:

    fullres_model = models.Sequential()

    fullres_model.add(AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))

    fullres_model.add(seg_model)

    fullres_model.add(UpSampling2D(IMG_SCALING))

else:

    fullres_model = seg_model

fullres_model.save('fullres_model.h5')
def raw_prediction(img, path=test_image_dir):

    c_img = imread(os.path.join(path, c_img_name))

    c_img = np.expand_dims(c_img, 0)/255.0

    cur_seg = fullres_model.predict(c_img)[0]

    return cur_seg, c_img[0]



def smooth(cur_seg):

    return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))



def predict(img, path=test_image_dir):

    cur_seg, c_img = raw_prediction(img, path=path)

    return smooth(cur_seg), c_img



## Get a sample of each group of ship count

samples = valid_df.groupby('ships').apply(lambda x: x.sample(1))

fig, m_axs = plt.subplots(samples.shape[0], 4, figsize = (15, samples.shape[0]*4))

[c_ax.axis('off') for c_ax in m_axs.flatten()]



for (ax1, ax2, ax3, ax4), c_img_name in zip(m_axs, samples.ImageId.values):

    first_seg, first_img = raw_prediction(c_img_name, train_image_dir)

    ax1.imshow(first_img)

    ax1.set_title('Image: ' + c_img_name)

    ax2.imshow(first_seg[:, :, 0], cmap=get_cmap('jet'))

    ax2.set_title('Model Prediction')

    reencoded = masks_as_color(multi_rle_encode(smooth(first_seg)[:, :, 0]))

    ax3.imshow(reencoded)

    ax3.set_title('Prediction Masks')

    ground_truth = masks_as_color(masks.query('ImageId=="{}"'.format(c_img_name))['EncodedPixels'])

    ax4.imshow(ground_truth)

    ax4.set_title('Ground Truth')

    

fig.savefig('validation.png')
test_paths = np.array(os.listdir(test_image_dir))

print(len(test_paths), 'test images found')
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
## let's see what we got

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
sub1 = pd.read_csv('../input/sample_submission_v2.csv')

sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])

sub1['EncodedPixels'] = None

print(len(sub1), len(sub))



sub = pd.concat([sub, sub1])

print(len(sub))

sub.to_csv('submission.csv', index=False)

sub.head()