import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from skimage.io import imread, imshow, imread_collection, concatenate_images
import matplotlib.pyplot as plt
%matplotlib inline

TRAIN_DIR = Path('../input/data-science-bowl-2018/stage1_train')
TEST_DIR = Path('../input/data-science-bowl-2018/stage1_test')

IMG_TYPE = '.png'         # Image type
IMG_CHANNELS = 3          # Default number of channels
IMG_DIR_NAME = 'images'   # Folder name including the image
MASK_DIR_NAME = 'masks'   # Folder name including the masks

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=FutureWarning, module='skimage')
from scipy import ndimage as ndi
from skimage.morphology import erosion, square

def read_image(observation_id, directory):
    return imread(sorted((directory / observation_id).glob('images/*.png'))[0])

def read_masks(observation_id, directory):
    return imread_collection(sorted((directory / observation_id).glob('masks/*.png')))

def segment_mask(masks):
    '''Combine a list of masks into a single image.'''
    mask = np.sum(masks, axis=0)
    return np.clip(mask, 0, 1).astype(np.uint8)

def segment_soft_mask(masks):
    '''
    EXPERIMENTAL
    Try a soft encoding for masks (as opposed to a 0/1 hard encoding) where the probability of a 
    mask is a function of the distance from the center of the nearest nucleus.
    
    RESULT
    This didn't end up working out too well, the masks were way too small and required tuning of
    the cutoff parameter. 
    '''
    final_mask = np.zeros(masks[0].shape) # pixel locations with a value of 0 denote the background
    for i, mask in enumerate(masks):
        distance = ndi.distance_transform_edt(mask)
        final_mask = np.maximum(final_mask, distance)
    return final_mask / np.max(final_mask)

def segment_eroded_mask(masks, size=2):
    '''Remove pixels at the boundary of a mask. Useful for ensuring that no two masks are touching.'''
    masks = [erosion(mask, square(size)) for mask in masks]
    mask = np.sum(masks, axis=0)
    return np.clip(mask, 0, 1).astype(np.uint8)

def instance_mask(masks):
    '''Returns an overlay where each instance location is labeled by an integer starting at 1 and incresasing.'''
    all_labels = np.zeros(masks[0].shape) # pixel locations with a value of 0 denote the background
    for i, mask in enumerate(masks):
        mask = mask > 0
        label = (mask)*(i+1) # pixel locations with a value of i denote the ith mask
        all_labels = np.maximum(all_labels, label) # for overlapping masks, use the higher value - this shouldn't ever happen for this dataset
    return all_labels.astype(np.uint8)

def separate_instances(label_image):
    '''
    Input: Labeled pixel map where each integer corresponds with one nucleus. 
    Returns: A list of masks where each mask shows the complete pixel mapping for one nucleus.
    '''
    all_masks = []
    for i in range(1, np.max(label_image)+1):
        mask = (label_image == i).astype(np.uint8)
        all_masks.append(mask)
    return all_masks
experimental_conditions = pd.read_csv('../input/dsb-observation-types/classes.csv')
experimental_conditions.head()
experimental_conditions.groupby(['foreground', 'background'])['background'].agg('count')
observations = os.listdir(TRAIN_DIR)
print(f'{len(observations)} training examples were found in {TRAIN_DIR}')
print(f'{len(os.listdir(TEST_DIR))} training examples were found in {TEST_DIR}')
train_observations = observations[:-60]
val_observations = observations[-60:]
sample = train_observations[0]

# for each observation, images/ contains one photo
image_files = sorted((TRAIN_DIR / sample).glob('images/*.png'))
print(f"Files found in 'images/': \n{image_files}")

# insepecting an example of an image
image = imread(image_files[0])
imshow(image)

print(f'\nImage dimensions: {image.shape}')
# for each observation, masks/ contains n masks where n is the number of nuclei identified in the image
mask_files = sorted((TRAIN_DIR / sample).glob('masks/*.png'))
print(f"Files found in 'masks/': \n{mask_files}")

# insepcting an example of the masks
masks = imread_collection(sorted((TRAIN_DIR / observations[0]).glob('masks/*.png')))
imshow(np.sum(masks, axis=0).astype(np.uint8)) # combine masks 

print(f'\nNumber of masks: {len(masks)}')
from skimage.morphology import label
from skimage.color import label2rgb

# treat masks as a segmentation problem, then use skimage.morphology.label to identify the instances
label_image = label(np.sum(masks, axis=0))

image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(image_label_overlay)
masks = separate_instances(label_image)

# Show the first 8 masks in a labeled image
fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(14, 8))
fig.suptitle("Masks for image", fontsize=16)
for i in range(min(len(masks), 8)):
    ax[i // 4, i % 4].imshow(masks[i])
for x in ax.ravel(): x.axis("off")
from skimage.measure import regionprops
import matplotlib.patches as mpatches
import random

def show_random_observations(observations, n=3, bboxes=False):
    fig, ax = plt.subplots(ncols=2, nrows=n, figsize=(20, n*6))

    for i in range(n):
        # Load an example image
        sample = random.choice(observations)
        image = read_image(sample, TRAIN_DIR)
        masks = instance_mask(read_masks(sample, TRAIN_DIR))
        image_label_overlay = label2rgb(masks, image=image, bg_label=0)

        ax[i, 0].imshow(image)
        ax[i, 0].set_title('input image')

        ax[i, 1].imshow(image_label_overlay)
        ax[i, 1].set_title('target labels')

        # also show bounding boxes just for fun
        if bboxes:
            for region in regionprops(masks):
                # take regions with large enough areas
                if region.area >= 1:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='red', linewidth=2)
                    ax[i, 1].add_patch(rect)

        print(sample) # in case I want to make a note of a specific observation
show_random_observations(train_observations, n=4)
from collections import namedtuple

# let's get some basic stats about our dataset 
Summary = namedtuple('Summary', ['observation_id', 'image_size', 'n_masks'])
train_data_summary = []

# create data frame of observation_id, image_size, n_masks
for observation_id in train_observations:
    train_data_summary.append(Summary(observation_id=observation_id,
                                      image_size=read_image(observation_id, TRAIN_DIR).shape,
                                      n_masks=len(read_masks(observation_id, TRAIN_DIR))
                                     )
                             )
    
df = pd.DataFrame(train_data_summary, columns=Summary._fields)
df.head()
df['image_size'].value_counts()
df['n_masks'].plot(kind='hist')
def encode_target(masks, w0=5, sigma=2):
    # ref : https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34/notebook
    masks = [erosion(mask, square(2)) for mask in masks]
    
    merged_mask = segment_mask(masks)
    weight = np.zeros(merged_mask.shape)
    # calculate weight for important pixels
    distances = np.array([ndi.distance_transform_edt(m==0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell 
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

    weight = w0 * np.exp(-(d1+d2)**2/(2*sigma**2)).astype(np.float32)
    weight = 1 + (merged_mask == 0) * weight
    return merged_mask - weight

def decode_target(encoding):
    target_mask = np.array(encoding == 0, dtype=np.uint8)
    weights = (-1 * encoding) + target_mask
    
    return target_mask, weights
encoding = encode_target(read_masks(train_observations[12], TRAIN_DIR))
mask, weight = decode_target(encoding)

fig, ax = plt.subplots(ncols=3, figsize=(16,8))

ax[0].imshow(mask) 
ax[0].set_title('Semantic segmentation')

ax[1].imshow(weight) 
ax[1].set_title('Separation weights')

ax[2].imshow(mask + weight) 
ax[2].set_title('Weights imposed on segmentation mask')
from keras.losses import binary_crossentropy
import keras.backend as K

def weighted_binary_crossentropy(y_true, y_pred):
    '''
    Calculates the weighted pixel-wise binary cross entropy. Expects target to be encoded as `(mask - weights)`. 
    '''
    # mask <- where value==0
    target_mask = K.cast(K.equal(y_true, 0), 'float32') 
    
    # weights calculated as described above
    weights = (-1 * y_true) + target_mask
    
    cce = binary_crossentropy(target_mask, y_pred)  
    wcce = cce * K.squeeze(weights, axis=-1)
    return K.mean(wcce, axis=-1)
def prepare_target(observation):
    masks = read_masks(observation, TRAIN_DIR)
    masks_resized = [cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA) for mask in masks]
    encoding = encode_target(masks_resized)
    return encoding
import cv2

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

# images
x_train = [read_image(observation, TRAIN_DIR) for observation in train_observations]
x_train = [cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA) for image in x_train]
x_train = np.array(x_train, dtype=np.uint8)
x_train = x_train[:,:,:,:IMG_CHANNELS]

x_val = [read_image(observation, TRAIN_DIR) for observation in val_observations]
x_val = [cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA) for image in x_val]
x_val = np.array(x_val, dtype=np.uint8)
x_val = x_val[:,:,:,:IMG_CHANNELS]

# targets
y_train = [prepare_target(observation) for observation in train_observations]
y_train = np.array(y_train)
y_train = np.expand_dims(y_train, axis=-1)

y_val = [prepare_target(observation) for observation in val_observations]
y_val = np.array(y_val)
y_val = np.expand_dims(y_val, axis=-1)
ix = 12

encoding = y_train[ix]
mask, weights = decode_target(encoding)
fig, ax = plt.subplots(ncols=3, figsize=(16,8))
ax[0].imshow(x_train[ix])
ax[0].set_title('Image')
ax[1].imshow(np.squeeze(mask))
ax[1].set_title('Mask')
ax[2].imshow(np.squeeze(weights))
ax[2].set_title('Weights')
from keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(horizontal_flip=True, 
                     vertical_flip=True)

image_datagen = ImageDataGenerator(rescale=1./255, **data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


# Provide the same seed and keyword arguments to the flow methods
seed = 1
batch_size = 4

# ------ training data ------
train_image_generator = image_datagen.flow(x_train, batch_size=batch_size, seed=seed)
train_mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)
train_steps = np.ceil(len(x_train) / batch_size)

# ------ validation data ------
val_image_generator = image_datagen.flow(x_val, batch_size=batch_size, seed=seed)
val_mask_generator = mask_datagen.flow(y_val, batch_size=batch_size, seed=seed)

# combine generators into one which yields image and masks
val_generator = zip(val_image_generator, val_mask_generator)
val_steps = np.ceil(len(x_val) / batch_size)
image, encoding = next(train_generator)
mask, weights = decode_target(encoding)

fig, ax = plt.subplots(ncols=3, figsize=(16, 8))
ax[0].imshow(image[0])
ax[1].imshow(np.squeeze(mask[0]))
ax[2].imshow(np.squeeze(weights[0]))
# define the u-net
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.core import Lambda, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import backend as K

import tensorflow as tf
def conv_block(inputs, filters, filter_size=3, drop_prob=0.2, regularizer=regularizers.l2(0.0001)):
    x = Conv2D(filters, filter_size, padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, filter_size, padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(drop_prob)(x)
    return x

def downsample(block):
    x = MaxPooling2D(pool_size=(2, 2)) (block)
    return x

def upsample(block, skip_connection, filters, regularizer=regularizers.l2(0.0001)):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizer)(block)
    stack = concatenate([skip_connection, x])
    return stack
from keras.optimizers import SGD

def build_unet(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3, drop_prob=0.2):
    
    regularizer=regularizers.l2(0.0001)
    
    # ---- Model ----
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Downsample
    encode_1 = conv_block(inputs, 16, regularizer=regularizer)
    down_1 = downsample(encode_1)
    
    encode_2 = conv_block(down_1, 32, regularizer=regularizer)
    down_2 = downsample(encode_2)
    
    encode_3 = conv_block(down_2, 64, regularizer=regularizer)
    down_3 = downsample(encode_3)
    
    encode_4 = conv_block(down_3, 128, regularizer=regularizer)
    down_4 = downsample(encode_4)
    
    bridge = conv_block(down_4, 256, regularizer=regularizer)
    
    up_4 = upsample(bridge, encode_4, 128)
    decode_4 = conv_block(up_4, 128, regularizer=regularizer)

    up_3 = upsample(decode_4, encode_3, 64)
    decode_3 = conv_block(up_3, 64, regularizer=regularizer)

    up_2 = upsample(decode_3, encode_2, 32)
    decode_2 = conv_block(up_2, 32, regularizer=regularizer)

    up_1 = upsample(decode_2, encode_1, 16)
    decode_1 = conv_block(up_1, 16, regularizer=regularizer)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizer)(decode_1)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=weighted_binary_crossentropy)
    return model

model = build_unet()
model.summary()
from keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.90, step_size=10)
checkpointer = ModelCheckpoint('unet_best.h5', verbose=1, save_best_only=True)

results = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=75, 
                              validation_data=val_generator, validation_steps=val_steps,
                              callbacks=[checkpointer, lr_sched])
model.load_weights('unet_best.h5')
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
def show_results(ix, model):
    '''Quick helper function to display predictions.''' 
    image = x_val[ix]
    key = val_observations[ix]

    masks = read_masks(key, TRAIN_DIR)
    masks_resized = [cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA) for mask in masks]
    target = label2rgb(instance_mask(masks_resized), image=image, bg_label=0)


    pred_mask = model.predict((x_val[ix]/255)[None])[0]
    pred_mask = np.squeeze((pred_mask > 0.5).astype(np.uint8))
    label_image = label(pred_mask)
    image_label_overlay = label2rgb(np.squeeze(label_image), image=image, bg_label=0)

    mask, weights = decode_target(np.squeeze(y_val[ix]))

    fig, ax = plt.subplots(2, 3, figsize=(20, 12))

    ax[0,0].imshow(image)
    ax[0,0].set_title('input')

    ax[1,0].imshow(weights)
    ax[1,0].set_title('loss weights')

    ax[0,1].imshow(mask)
    ax[0,1].set_title('semantic target')

    ax[0,2].imshow(target)
    ax[0,2].set_title('instance target')

    ax[1,1].imshow(pred_mask*1.0)
    ax[1,1].set_title('prediction: segmented mask')

    ax[1,2].imshow(image_label_overlay)
    ax[1,2].set_title('prediction: instance mask')

    for x in ax.ravel(): x.axis("off")
show_results(48, model)
show_results(54, model)
show_results(26, model)
# Kaggle Kernels have limited memory, so we'll remove this model from memory
import gc
del model 
gc.collect()
def dense_block(stack, n_layers, growth_rate, filter_size=3, drop_prob=0.2):
    
    for layer in range(n_layers):
        x = BatchNormalization()(stack)
        x = Activation('relu')(x)
        x = Conv2D(growth_rate, filter_size, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
        x = SpatialDropout2D(drop_prob)(x)
        stack = concatenate([stack, x])
        
    return stack

def downsample(block, n_filters, drop_prob=0.2):
    x = BatchNormalization()(block)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = SpatialDropout2D(drop_prob)(x)
    x = MaxPooling2D(pool_size=(2, 2)) (x)
    return x

def upsample(block, skip_connection, n_filters):
    x = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.0001)) (block)
    stack = concatenate([skip_connection, x])
    return stack
def build_fcdensenet(m, IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=3):
    '''
    Keras Implementation of the Fully Convolutional DenseNet. 
    m: List containing the number of feature maps for each dense block. 
    '''
    regularizer = regularizers.l2(0.0001)
    
    # ---- Model ----
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    conv = Conv2D(m[0], (3,3), padding='same', kernel_regularizer=regularizer) (inputs)

    encode_1 = dense_block(conv, 4, 12)
    down_1 = downsample(encode_1, m[1])

    encode_2 = dense_block(down_1, 4, 12)
    down_2 = downsample(encode_2, m[2])

    encode_3 = dense_block(down_2, 4, 12)
    down_3 = downsample(encode_3, m[3])

    encode_4 = dense_block(down_3, 4, 12)
    down_4 = downsample(encode_4, m[4])

    bridge = dense_block(down_4, 4, 12)

    up_4 = upsample(bridge, encode_4, m[6])
    decode_4 = dense_block(up_4, 4, 12)

    up_3 = upsample(decode_4, encode_3, m[7])
    decode_3 = dense_block(up_3, 4, 12)

    up_2 = upsample(decode_3, encode_2, m[8])
    decode_2 = dense_block(up_2, 4, 12)

    up_1 = upsample(decode_2, encode_1, m[9])
    decode_1 = dense_block(up_1, 4, 12)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizer) (decode_1)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=weighted_binary_crossentropy)
    return model
# determine number of feature maps for each layer, following guidance from the original paper
m = [32]
encoder_blocks = 5 # 4 dense + bridge
decoder_blocks = 4
for i in range(encoder_blocks):
    m.append(m[i] + 4*12)
    
for i in range(decoder_blocks):
    m.append(m[4-i] + 4*12 + 4*12) # skip connection + feature maps from the upsampled block + feature maps in the new block

dense_model = build_fcdensenet(m)
dense_model.summary()
lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.90, step_size=10)
checkpointer = ModelCheckpoint('fcdense_best.h5', verbose=1, save_best_only=True)

results = dense_model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=75, 
                                   validation_data=val_generator, validation_steps=val_steps,
                                   callbacks=[checkpointer, lr_sched])
dense_model.load_weights('fcdense_best.h5')
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
show_results(48, dense_model)
show_results(54, dense_model)
show_results(26, dense_model)
def iou_at_thresholds(target_mask, pred_mask, thresholds=np.arange(0.5,1,0.05)):
    '''Returns True if IoU is greater than the thresholds.'''
    intersection = np.logical_and(target_mask, pred_mask)
    union = np.logical_or(target_mask, pred_mask)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    return iou > thresholds

def calculate_iou_tensor(target_masks, pred_masks, thresholds=np.arange(0.5,1,0.05)):
    iou_tensor = np.zeros([len(thresholds), len(pred_masks), len(target_masks)])

    # TODO: Use tiling to make this faster
    for i, p_mask in enumerate(pred_masks):
        for j, t_mask in enumerate(target_masks):
            iou_tensor[:, i, j] = iou_at_thresholds(t_mask, p_mask, thresholds)

    return iou_tensor

def calculate_average_precision(target_masks, pred_masks, thresholds=np.arange(0.5,1,0.05)):
    '''Calculates the average precision over a range of thresholds for one observation (with a single class).'''
    iou_tensor = calculate_iou_tensor(target_masks, pred_masks, thresholds=thresholds)
    
    TP = np.sum((np.sum(iou_tensor, axis=2) == 1), axis=1)
    FP = np.sum((np.sum(iou_tensor, axis=1) == 0), axis=1)
    FN = np.sum((np.sum(iou_tensor, axis=2) == 0), axis=1)

    precision = TP / (TP + FP + FN)

    return np.mean(precision)

def calculate_mean_average_precision(y_true, y_pred):
    '''
    # Arguments
        y_true: A list of lists each containing the target masks for a given observation.
        y_pred: A list of lists each containing the predicted masks for a given observation.
    '''
    average_precision = []
    thresholds=np.arange(0.5,1,0.05)
    
    for target, prediction in zip(y_true, y_pred):
        ap = calculate_average_precision(target, prediction, thresholds=thresholds)
        average_precision.append(ap)
        
    return average_precision
# resize individual masks for final evaluation
def resize_masks(masks):
    return [cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA) for mask in masks]

y_val_masks = [read_masks(observation, TRAIN_DIR) for observation in val_observations]
y_val_masks = [resize_masks(masks) for masks in y_val_masks]
from skimage.morphology import dilation

def separate_instances_with_tricks(label_image):
    pred_masks = []
    for i in range(1, np.max(label_image)+1):
        mask = (label_image == i).astype(np.uint8)
        if np.sum(mask) > 5:
            dilated_mask = dilation(mask, square(3))
            pred_masks.append(mask)
    return pred_masks

preds = dense_model.predict(x_val/255)
preds = np.squeeze((preds > 0.5).astype(np.uint8))
preds_masks = [label(pred) for pred in preds]
preds_masks = [separate_instances_with_tricks(label) for label in preds_masks]
avg_prec = calculate_mean_average_precision(y_val_masks, preds_masks)
import seaborn as sns
sns.distplot(avg_prec)
print(f'Mean average precision: {np.mean(avg_prec)}')
test_data = {}

# read in the data
for observation in os.listdir(TEST_DIR):
    test_data[observation] = {'image': read_image(observation, TEST_DIR),
                              'size': read_image(observation, TEST_DIR).shape[:2]}
x_test = [observation['image'] for observation in test_data.values()]
x_test = [cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)[:,:,:IMG_CHANNELS] for image in x_test]
x_test = np.array(x_test)
sizes = [data['size'] for data in test_data.values()]

preds = dense_model.predict(x_test/255)
preds = [cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) for image, size in zip(preds, sizes)]
preds_masks = [label(pred > 0.5) for pred in preds]
preds_masks = [separate_instances(label) for label in preds_masks]
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
from collections import namedtuple

Mask = namedtuple('Mask', ['observation_id', 'rle'])
rle_preds = []

for _id, preds in zip(test_data.keys(), preds_masks):
    for pred in preds:
        if np.sum(pred) > 10:
            rle_preds.append(Mask(observation_id=_id, 
                                  rle=rle_encoding(pred)
                                 )
                            )

rle_df = pd.DataFrame(rle_preds, columns=['ImageId', 'EncodedPixels'])
rle_df.head()
rle_df.to_csv('submission.csv', index=False)