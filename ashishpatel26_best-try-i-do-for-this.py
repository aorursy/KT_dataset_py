import math

import bcolz
import numpy as np
from skimage.transform import resize

import tensorflow as tf
from keras import backend as K


OR_IM_WIDTH = 101
OR_IM_HEIGHT = 101
OR_IM_CHANNEL = 3

IM_WIDTH = 128
IM_HEIGHT = 128
IM_CHAN = 1


def save_arr (fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    

def load_array(fname):
    return bcolz.open(fname)[:]


def upsample(img):
    return resize(img, (IM_HEIGHT, IM_WIDTH, IM_CHAN), mode='constant', preserve_range=True)

    
def downsample(img):
    return resize(img, (OR_IM_HEIGHT, OR_IM_WIDTH), mode='constant', preserve_range=True)


def rle_decode(rle, shape):
    """
    rle: run-length string or list of pairs of (start, length)
    shape: (height, width) of array to return 
    Returns
    -------
        np.array: 1 - mask, 0 - background
    """
    if isinstance(rle, float) and math.isnan(rle):
        rle = []
    if isinstance(rle, str):
        rle = [int(num) for num in rle.split(' ')]
    # [0::2] means skip 2 since 0 until the end - list[start:end:skip]
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    img = img.reshape(1, shape[0], shape[1])
    img = img.T
    return img


def rle_encode(img):
    """
    img: np.array: 1 - mask, 0 - background
    Returns
    -------
    run-length string of pairs of (start, length)
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle if rle else float('nan')


def iou(y_true, y_pred):
    """ Intersection over Union Metric
    """
    component1 = y_true.astype(dtype=bool)
    component2 = y_pred.astype(dtype=bool)

    overlap = component1 * component2 # Logical AND
    union = component1 + component2 # Logical OR

    iou = overlap.sum() / float(union.sum())
    return iou


def iou_batch(y_true, y_pred):
    batch_size = y_true.shape[0]
    metric = []
    for i in range(batch_size):
        value = iou_metric(y_true[i], y_pred[i])
        metric.append(value)
    return np.mean(metric)


def mean_iou(y_true, y_pred):
    """Keras valid metric to use with a keras.Model
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

from utils import *

import os
import glob
import random
import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.transform import resize
from keras.preprocessing import image as image_utils
train_path = "../input/train/images/"
train_masks_path = "../input/train/masks/"
test_path = "../input/test/images"
train_files = sorted(glob.glob(os.path.join(train_path, "*.png")))
masks_files = sorted(glob.glob(os.path.join(train_masks_path, "*.png")))
test_files = sorted(glob.glob(os.path.join(test_path, "*.png")))
assert len(train_files) == len(masks_files)
ids_train  = []
X_train = np.zeros((len(train_files), OR_IM_HEIGHT, OR_IM_WIDTH, OR_IM_CHANNEL), dtype=np.uint8)
y_train = np.zeros((len(masks_files), OR_IM_HEIGHT, OR_IM_WIDTH, OR_IM_CHANNEL), dtype=np.uint8)
X_train.shape, y_train.shape
for i, (train_path, mask_path) in tqdm.tqdm_notebook(enumerate(zip(train_files, masks_files)), total=len(train_files)):
    train_id = os.path.basename(train_path)[:-4]
    mask_id = os.path.basename(mask_path)[:-4]
    assert train_id == mask_id
    ids_train.append(train_id)
    
    x = image_utils.img_to_array(image_utils.load_img(train_path))
    X_train[i] = x

    y = image_utils.img_to_array(image_utils.load_img(mask_path))
    y_train[i] = y
len(ids_train), X_train.shape, y_train.shape
n_images = 6
fig, axarr = plt.subplots(2, n_images, figsize=(15, 5))
for image in range(n_images):
    n = random.randint(1, X_train.shape[0])
    axarr[0, image].imshow(X_train[n])
    axarr[1, image].imshow(y_train[n])
fig.tight_layout()
ids_test = []
X_test = np.zeros((len(test_files), OR_IM_HEIGHT, OR_IM_WIDTH, OR_IM_CHANNEL), dtype=np.uint8)

for i, test_path in tqdm.tqdm_notebook(enumerate(test_files), total=len(test_files)):
    test_id = os.path.basename(test_path)[:-4]
    ids_test.append(test_id)
    
    x = image_utils.img_to_array(image_utils.load_img(test_path))
    X_test[i] = x
len(ids_test), X_test.shape
n_images = 6
fig, axarr = plt.subplots(1, n_images, figsize=(15, 5))
for image in range(n_images):
    n = random.randint(1, X_test.shape[0])
    axarr[image].imshow(X_test[n])
fig.tight_layout()
coverage_train = np.zeros((X_train.shape[0], ), dtype=np.float64)
for i, (image, mask) in tqdm.tqdm_notebook(enumerate(zip(X_train, y_train)), total=X_train.shape[0]):
    coverage_train[i] = np.mean(mask) / 255
coverage_train
strata_train = np.zeros((X_train.shape[0], ), dtype=np.uint8)
def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
v_cov_to_class = np.vectorize(cov_to_class)
strata_train = v_cov_to_class(coverage_train)
strata_train
fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(coverage_train, kde=False, ax=axs[0])
sns.distplot(strata_train, bins=10, kde=False, ax=axs[1])
plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class")
n_images = 11
fig, axarr = plt.subplots(2, n_images, figsize=(18, 3))
for image in range(n_images):
    statum_img = X_train[strata_train == image]
    statum_mask = y_train[strata_train == image]
    n = random.randint(1, statum_img.shape[0])
    axarr[0, image].imshow(statum_img[n])
    axarr[1, image].imshow(statum_mask[n])
fig.tight_layout()
X_train_up = np.array([upsample(img) for img in tqdm.tqdm_notebook(X_train, total=X_train.shape[0])])
y_train_up = np.array([upsample(img) for img in tqdm.tqdm_notebook(y_train, total=y_train.shape[0])])
X_test_up = np.array([upsample(img) for img in tqdm.tqdm_notebook(X_test, total=X_test.shape[0])])
X_train_up.shape, y_train_up.shape, X_test_up.shape
# save_arr("ids_train", ids_train)
# save_arr("X_train", X_train_up)
# save_arr("y_train", y_train_up)
# save_arr("strata_train", strata_train)
# save_arr("ids_test", ids_test)
# save_arr("X_test", X_test_up)

# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
random_state = 42
ids_train_ = ids_train
X_train_ = X_train_up
y_train_ = y_train_up.astype(np.bool)
strata_train = strata_train
ids_test = ids_test
X_test = X_test_up
im_width  = X_train_.shape[1]
im_height = X_train_.shape[2]
im_chan = X_train_.shape[3]
X_train_.shape, y_train_.shape, im_width, im_height, im_chan
X_train_  = np.append(X_train_, [np.fliplr(x) for x in X_train_], axis=0)
y_train_ = np.append(y_train_, [np.fliplr(x) for x in y_train_], axis=0)
strata_train = np.append(strata_train, strata_train)
X_train_.shape, y_train_.shape, strata_train.shape
from sklearn.model_selection import train_test_split
random_state  = 42
X_train, X_valid, y_train, y_valid = train_test_split(X_train_,y_train_,test_size=0.2, stratify=strata_train, random_state=random_state)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
def build_model(input_layer, start_neurons):
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    uncov1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer
input_layer  = Input((im_height, im_width, im_chan))
output_layer = build_model(input_layer, 16)
model = Model(input_layer , output_layer)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', mean_iou])
model.summary()
early_stopping = EarlyStopping(patience=10, verbose=1)
model_checkpoint = ModelCheckpoint("unet-dropout.model", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

epochs = 100
batch_size = 16
history = model.fit(X_train, y_train,
                    validation_data=[X_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr])
pred_test = model.predict(X_test, verbose=1)
threshold = 0.5
pred_test_tresh = np.int32(pred_test > threshold)
from utils import *
import tqdm
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from skimage.transform import resize
import pandas as pd
threshold = 0.5
pred_test_tresh = np.int32(pred_test > threshold)
preds_test_downsample  = []
for i in tqdm.tnrange(len(pred_test)):
    # Resize it back to original size: 101x101
    preds_test_downsample.append(np.int32(downsample(pred_test[i]) > threshold))
pred_dict  = {img_id: rle_encode(preds_test_downsample[i]) for i, img_id in tqdm.tqdm_notebook(enumerate(ids_test), total=len(ids_test))}
sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.head()
sub.to_csv("submission.csv")
