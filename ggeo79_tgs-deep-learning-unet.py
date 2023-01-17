# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import shutil
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import transform
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, SeparableConv2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.core import Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils, Sequence
from keras.optimizers import Adam, SGD, RMSprop
TRAIN_PATH = '../input/train/'
TEST_PATH = '../input/test/'
IMAGE_WIDTH_ORIG = 101
IMAGE_HEIGHT_ORIG = 101
IMAGE_WIDTH_TARGET = 128
IMAGE_HEIGHT_TARGET = 128
IMG_CHANNELS = 1
# Load train/depth ids
train_data = pd.read_csv('../input/train.csv', index_col = 'id', usecols=[0])
depths_data = pd.read_csv('../input/depths.csv', index_col = 'id')
train_data = train_data.join(depths_data)
# Read images/masks
train_data['images'] = [np.array(image.load_img(TRAIN_PATH + 'images/{}.png'.format(idx), color_mode='grayscale')).astype(np.float32) for idx in train_data.index]
train_data['masks'] = [np.array(image.load_img(TRAIN_PATH + 'masks/{}.png'.format(idx), color_mode='grayscale')).astype(np.float32) for idx in train_data.index]
# Rescale pixel values from range [0-255] to range [0-1] (neural networks prefer smaller values)
train_data['images'] /= 255
train_data['masks'] /= 255
# Compute salt coverage
# Every salt pixel has value 1 and every non salt has the value 0.So, we sum in  order to obtain the
# number of salt pixels and then we divide by the total number of pixels in the image
train_data = train_data.assign(coverage = train_data.masks.map(np.sum) / (IMAGE_WIDTH_ORIG * IMAGE_HEIGHT_ORIG))
# create salt coverage classes
def coverage_to_class(inp):
    for i in range(0, 11):
        if 10 * inp <= i:
            return i
        
train_data = train_data.assign(coverage_class = train_data.coverage.map(coverage_to_class))
# check coverage vs classes
plt.scatter(train_data.coverage, train_data.coverage_class)
plt.xlabel('Coverage')
plt.ylabel('Coverage class')
def data():
    # Resize and resample
    train_data_images_sampling = np.array([resize(r, (IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET), mode='constant', preserve_range=True) for r in train_data.images])
    train_data_images_sampling = train_data_images_sampling.reshape(-1, IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1)
    train_data_masks_sampling = np.array([resize(r, (IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET), mode='constant', preserve_range=True) for r in train_data.masks])
    train_data_masks_sampling = train_data_masks_sampling.reshape(-1, IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1)
    
    # split data into train and validation sets using the salt coverage as a stratification criterion
    # in order to have a more homogenous split 
    # note, that X_train, X_val, y_train, y_val will have the target size
    ids_train, ids_val, X_train, X_val, y_train, y_val , coverage_train, coverage_test, depths_train, depths_test = train_test_split(
                        train_data.index.values,
                        train_data_images_sampling, 
                        train_data_masks_sampling, 
                        train_data.coverage.values,
                        train_data.z.values,
                        test_size=0.2, stratify=train_data.coverage_class,random_state=1340)

    return ids_train, ids_val, X_train, X_val, y_train, y_val , coverage_train, coverage_test, depths_train, depths_test
# test data
test_data = depths_data.loc[~depths_data.index.isin(train_data.index)]
X_test = [np.array(image.load_img(TEST_PATH + 'images/{}.png'.format(idx), color_mode='grayscale')).astype(np.float32) / 255 for idx in test_data.index]
X_test = np.array([resize(r, (IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET), mode='constant', preserve_range=True) for r in X_test]).reshape(-1, IMAGE_HEIGHT_TARGET,IMAGE_WIDTH_TARGET, 1)
def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

# Create model
def build_UNet_model(X_train, y_train, X_val, y_val):
    
    concat_axis=3
    
    input_layer = Input((IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1))      
    
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis) 
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    ch, cw = get_crop_shape(input_layer, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
       
    output_layer = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(conv9)

    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint('./tgs_b_32_relu_optim_adam_UNET.model', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1)
    
    model = Model(input_layer, output_layer)
    adam = Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    history = model.fit(X_train,
                        y_train,
                        validation_data=[X_val, y_val],
                        batch_size=32,
                        epochs = 50,
                        callbacks=[model_checkpoint, early_stopping])
    
    return history,model
ids_train, ids_val, X_train, X_val, y_train, y_val , coverage_train, coverage_test, depths_train, depths_test = data()
#flip
X_train = np.append(X_train, [np.fliplr(f) for f in X_train], axis=0)
y_train = np.append(y_train, [np.fliplr(f) for f in y_train], axis=0)
# rotate
X_train = np.append(X_train, [transform.rotate(r, angle=45, mode='reflect') for r in X_train], axis=0)
y_train = np.append(y_train, [transform.rotate(r, angle=45, mode='reflect') for r in y_train], axis=0)
history, model = build_UNet_model(X_train, y_train, X_val, y_val)
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(20,4))
ax_loss.plot(history.epoch, history.history['loss'], label='Train loss')
ax_loss.plot(history.epoch, history.history['val_loss'], label='Val loss')
ax_loss.legend()
ax_acc.plot(history.epoch, history.history['acc'], label = 'Train acc')
ax_acc.plot(history.epoch, history.history['val_acc'], label = 'Val acc')
ax_acc.legend()
# Load the best saved model
model = load_model('./tgs_b_32_relu_optim_adam_UNET.model')
# predict on validation and test data
pred_val = model.predict(X_val)
pred_test = model.predict(X_test)
# convert result to 0 or 1
pred_val_int = (pred_val > 0.5).astype(np.uint8)
pred_test_int = (pred_test > 0.5).astype(np.uint8)
# Squeeze one dimension to be able to plot
X_train_squeeze = X_train.squeeze()
y_train_squeeze = y_train.squeeze()
pred_val_squeeze = pred_val.squeeze()
X_val_squeeze = X_val.squeeze()
y_val_squeeze = y_val.squeeze()
len(X_train)
max_images = 6

fig, axes = plt.subplots(3, max_images, figsize=(25, 14))

for i in  range(max_images):   
    idx = np.random.randint(0, len(X_val))
   
    axes[0][i].imshow(X_val_squeeze[idx], cmap='Greys')
    axes[1][i].imshow(y_val_squeeze[idx], cmap='Greens')
    axes[2][i].imshow(pred_val_squeeze[idx], cmap='YlOrRd') 

plt.suptitle('Top: Validation images - Middle: Validation masks  - Bottom: Predicted images')
# From kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
# metrics
# use the original masks for the metric 
y_valid_orig = np.array([train_data.loc[idx].masks for idx in ids_val])
# resize to the original size in order to compare
pred_val = pred_val.reshape(-1,IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET)
pred_val = np.array([resize(r, (IMAGE_HEIGHT_ORIG, IMAGE_WIDTH_ORIG), mode='constant', preserve_range=True) for r in pred_val])  
thresholds = np.linspace(0.1, 1, 40)
iou = np.array([iou_metric_batch(y_valid_orig, np.int32(pred_val > threshold)) for threshold in thresholds])
plt.plot(thresholds, iou)
plt.xlabel('threshold')
plt.ylabel('IoU')
plt.show()
best_threshold = thresholds[np.argmax(iou)]
best_threshold, max(iou)
# From https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
tmp = [resize((pred_test[i] > best_threshold), (IMAGE_HEIGHT_ORIG, IMAGE_WIDTH_ORIG), mode='constant', preserve_range=True) for i in range(len(test_data.index.values))]
pred_dict = {idx: RLenc(np.round(tmp[i])) for i, idx in enumerate(test_data.index.values)}
submissions = pd.DataFrame.from_dict(pred_dict, orient='index')
submissions.index.names = ['id']
submissions.columns = ['rle_mask']
submissions.to_csv('./submission_tgs_b_32_relu_optim_adam_UNET.model.csv')