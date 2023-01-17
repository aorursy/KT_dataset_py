#!pip install fastai

#!pip show fastai

#from fastai import *

#from fastai.conv_learner import *

#from fastai.dataset import *



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread

import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap

from skimage.segmentation import mark_boundaries

from skimage.util import montage

from skimage.morphology import binary_opening, disk, label

import gc; gc.enable() # memory is tight
train_image_dir = '../input/airbus-ship-detection/train_v2/'



SAMPLES_PER_GROUP = 40000

BATCH_SIZE = 48

VALID_IMG_COUNT = 900

# downsampling in preprocessing

IMG_SCALING = (3, 3)



# related to model fitting and prediction

# maximum number of steps_per_epoch in training

MAX_TRAIN_STEPS = 9

MAX_TRAIN_EPOCHS = 99

AUGMENT_BRIGHTNESS = False
from skimage.morphology import label

def multi_rle_encode(img):

    labels = label(img[:, :, 0])

    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]



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
data = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')

    

data.head()
print(type(data['EncodedPixels'][0]))

print(type(data['EncodedPixels'][2]))
data['ships'] = data['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

unique_img_ids = data.groupby('ImageId').agg({'ships': 'sum'}).reset_index()

unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)

#unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

#unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: os.stat('../input/airbus-ship-detection/train_v2/'+c_img_id).st_size/1024)

#unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]



data.drop(['ships'], axis=1, inplace=True)



unique_img_ids.head(5)
unique_img_ids['ships'].hist(bins=unique_img_ids['ships'].max())
balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

balanced_train_df['ships'].hist(bins=balanced_train_df['ships'].max()+1)

print(balanced_train_df.shape[0], 'masks')
from sklearn.model_selection import train_test_split

train_ids, valid_ids = train_test_split(balanced_train_df, 

                 test_size = 0.2, 

                 stratify = balanced_train_df['ships'])

train_df = pd.merge(data, train_ids)

valid_df = pd.merge(data, valid_ids)

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

print('make generater completed')
train_x, train_y = next(train_gen)

print('x', train_x.shape)

print('y', train_y.shape)
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)



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

valid_gen = make_image_gen(valid_df, VALID_IMG_COUNT)

valid_x, valid_y = next(valid_gen)

print(valid_x.shape, valid_y.shape)
gc.collect()
from keras import models, layers



input_img = layers.Input(train_x.shape[1:], name = 'RGB_Input')

pp_in_layer = input_img



#pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)

#pp_in_layer = layers.BatchNormalization()(pp_in_layer)



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



u5 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)

u5 = layers.concatenate([u5, c3])

c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u5)

c5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c5)



u6 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = layers.concatenate([u6, c2])

c6 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u6)

c6 = layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c6)



u7 = layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = layers.concatenate([u7, c1], axis=3)

c7 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u7)

c7 = layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c7)



d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c7)



seg_model = models.Model(inputs=[input_img], outputs=[d])

seg_model.summary()
import keras.backend as K

from keras.optimizers import Adam

from keras.losses import binary_crossentropy

import tensorflow as tf



sess = tf.compat.v1.Session()



## intersection over union

def IoU(y_true, y_pred, eps=1e-6):

    #if K.sum(y_true, axis=[1,2,3]) == 0.0:

    #    return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros

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

    #aug_gen = create_aug_gen(make_image_gen(train_df))

    loss_history = [seg_model.fit_generator(train_gen,

                                 steps_per_epoch=step_count,

                                 epochs=MAX_TRAIN_EPOCHS,

                                 validation_data=(valid_x, valid_y),

                                 callbacks=callbacks_list,

                                workers=1 # the generator is not very thread safe

                                           )]

    return loss_history



while True:

    loss_history = fit()

    if np.min([mh.history['val_loss'] for mh in loss_history]) < 0.2:#< -0.2

        break
seg_model.load_weights(weight_path)

#seg_model.save('seg_model.h5') # save for later use



if IMG_SCALING is not None:

    fullres_model = models.Sequential()

    fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))

    fullres_model.add(seg_model)

    fullres_model.add(layers.UpSampling2D(IMG_SCALING))

else:

    fullres_model = seg_model

fullres_model.save('fullres_model.h5')
test_image_dir = '../input/airbus-ship-detection/test_v2/'

def raw_prediction(img, path=test_image_dir):

    c_img = imread(os.path.join(path, img))

    c_img = np.expand_dims(c_img, 0)/255.0

    cur_seg = fullres_model.predict(c_img)[0]

    return cur_seg, c_img[0]



def predict(img, path=test_image_dir):

    cur_seg, c_img = raw_prediction(img, path=path)

    return cur_seg, c_img
from tqdm import tqdm_notebook



test_paths = np.array(os.listdir(test_image_dir))



def pred_encode(img):

    cur_seg, _ = predict(img)

    cur_rles = multi_rle_encode(cur_seg)

    return [[img, rle] for rle in cur_rles if rle is not None]



out_pred_rows = []

for c_img_name in tqdm_notebook(test_paths): ## only a subset as it takes too long to run

    out_pred_rows += pred_encode(c_img_name)
print(out_pred_rows)

sub = pd.DataFrame([['*.jpg', '1 2']] + out_pred_rows)

sub.columns = ['ImageId', 'EncodedPixels']

sub = sub.drop(0,0)

sub = sub[sub.EncodedPixels.notnull()]

sub.head()
sub1 = pd.read_csv('../input/airbus-ship-detection/sample_submission_v2.csv')

sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])

sub1['EncodedPixels'] = None

print(len(sub1), len(sub))



sub = pd.concat([sub, sub1])

print(len(sub))

sub.to_csv('submission1.csv', index=False)

sub.head()