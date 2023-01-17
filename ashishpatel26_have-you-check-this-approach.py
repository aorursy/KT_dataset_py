# References : https://www.kaggle.com/bguberfain/unet-with-depth/notebook
# https://www.kaggle.com/stefanie04736/simple-keras-model-with-k-fold-cross-validation

import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# Set some parameters
im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth
path_train = '../input/train/'
path_test = '../input/test/'

df_depths = pd.read_csv('../input/depths.csv', index_col='id')
df_depths.head()
train_ids = next(os.walk(path_train+"images"))[2]
test_ids = next(os.walk(path_test+"images"))[2]
# Get and resize train images and masks
X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    
    # Depth
    X_feat[n] = df_depths.loc[id_.replace('.png', ''), 'z']
    
    # Load X
    img = load_img(path + '/images/' + id_, grayscale=True)
    x_img = img_to_array(img)
    x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)
    
    # Create cumsum x
    x_center_mean = x_img[border:-border, border:-border].mean()
    x_csum = (np.float32(x_img)-x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    # Load Y
    mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
    mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

    # Save images
    X[n, ..., 0] = x_img.squeeze() / 255
    X[n, ..., 1] = x_csum.squeeze()
    y[n] = mask / 255

print('Done!')
np.sum(y)/y.shape[0]
from sklearn.model_selection import StratifiedShuffleSplit
# Split train and valid
X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(X, X_feat, y, test_size=0.15, random_state=42)
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.16, random_state=42) # Want a balanced split for all the classes
# for train_index, trainxfeat_index, test_index in sss.split(X,X_feat,y):
#     print("Using {} for training,{} for festing and {} for validation".format(len(train_index), len(trainxfeat_index), len(test_index)))
#     x_train, x_valid = X[train_index], X[test_index]
#     X_feat_train, X_feat_valid = X_feat[trainxfeat_index], X_feat[trainxfeat_index]
#     y_train, y_valid = y[train_index], y[test_index]
# Normalize X_feat
x_feat_mean = X_feat_train.mean(axis=0, keepdims=True)
x_feat_std = X_feat_train.std(axis=0, keepdims=True)
X_feat_train -= x_feat_mean
X_feat_train /= x_feat_std

X_feat_valid -= x_feat_mean
X_feat_valid /= x_feat_std
def get_model():
    # Build U-Net model
    input_img = Input((128,128,2), name='img')
    input_features = Input((n_features, ), name='feat')

    c1 = Conv2D(32, (3, 3), activation='relu', padding='same') (input_img)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same') (c1)
    p1 = AveragePooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same') (c2)
    p2 = AveragePooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (c3)
    p3 = AveragePooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (c4)
    p4 = AveragePooling2D(pool_size=(2, 2)) (c4)

    # Join features information in the depthest layer
    f_repeat = RepeatVector(8*8)(input_features)
    f_conv = Reshape((8, 8, n_features))(f_repeat)
    p4_feat = concatenate([p4, f_conv], -1)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4_feat)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[input_img, input_features], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy') #, metrics=[mean_iou]) # The mean_iou metrics seens to leak train and test values...
    return model

model = get_model()
model.summary()
from keras.preprocessing.image import ImageDataGenerator
batch_size=40
# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip = False,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=666)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

# Finally create generator
gen_flow = gen_flow_for_two_inputs(X_train, X_feat_train, y_train)

def get_callbacks():
    mcp_save =  ModelCheckpoint('model-tgs-salt-1.h5', save_best_only=True, monitor='val_loss', verbose=1)
    earlystopping = EarlyStopping(patience=10, verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4)
    return [mcp_save, reduce_lr_loss, earlystopping]

# batch_size=16
# generator = gen.flow({'img': X_train, 'feat': X_feat_train}, y_train, batch_size = batch_size)
# val_generator = gen.flow({'img': X_train, 'feat': X_feat_train}, y_train, batch_size = batch_size)
callbacks = get_callbacks()    
# results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks,
#                     validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))

# Fit the model using our generator defined above
result = model.fit_generator(gen_flow, validation_data=([X_valid, X_feat_valid], y_valid),
                    steps_per_epoch=len(X_train) / batch_size, epochs=60,callbacks=callbacks)

# for j, train_idx in enumerate(folds):
#     print('\nFold ',j)
#     callbacks = get_callbacks()
#     generator = gen.flow({'img': X_train, 'feat': X_feat_train}, y_train, batch_size = batch_size)
#     val_generator = gen.flow({'img': X_train, 'feat': X_feat_train}, y_train, batch_size = batch_size)
#     model = get_model()
#     model.fit_generator(
#                 generator,
#                 steps_per_epoch=len(X_train)/batch_size,
#                 epochs=30,
#                 shuffle=True,
#                 verbose=1,
#                 validation_data = val_generator,
#                 callbacks = callbacks)
    
#     print(model.evaluate(X_valid_cv, y_valid_cv))

#     results = model.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=16, epochs=50, callbacks=callbacks,
#                     validation_data=({'img': X_valid, 'feat': X_feat_valid}, y_valid))  
#     print(model.evaluate({'img': X_valid, 'feat': X_feat_valid}, y_valid))
    
#     model.fit_generator(
#                 generator,
#                 steps_per_epoch=len(X_train_cv)/batch_size,
#                 epochs=15,
#                 shuffle=True,
#                 verbose=1,
#                 validation_data = (X_valid_cv, y_valid_cv),
#                 callbacks = callbacks)




# for j, (train_idx, val_idx) in enumerate(folds):
    
#     print('\nFold ',j)
#     X_train_cv = X_train[train_idx]
#     y_train_cv = y_train[train_idx]
#     X_valid_cv = X_train[val_idx]
#     y_valid_cv= y_train[val_idx]
    
#     name_weights = "final_model_fold" + str(j) + "_weights.h5"
#     callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
#     generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)
#     model = get_model()
#     model.fit_generator(
#                 generator,
#                 steps_per_epoch=len(X_train_cv)/batch_size,
#                 epochs=15,
#                 shuffle=True,
#                 verbose=1,
#                 validation_data = (X_valid_cv, y_valid_cv),
#                 callbacks = callbacks)
    
#     print(model.evaluate(X_valid_cv, y_valid_cv))


# [
#     ModelCheckpoint('model-tgs-salt-1.h5', save_best_only=True, monitor='val_loss', verbose=1)
#     EarlyStopping(patience=10, verbose=1),
# #     ReduceLROnPlateau(patience=3, verbose=1),
#     ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4),
# #     ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True, save_weights_only=True)
#     ]
# Get and resize test images
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.float32)
X_feat_test = np.zeros((len(test_ids), n_features), dtype=np.float32)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
    path = path_test
    
    # Depth
    X_feat_test[n] = df_depths.loc[id_.replace('.png', ''), 'z']
    
    # Load X
    img = load_img(path + '/images/' + id_, grayscale=True)
    x = img_to_array(img)
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    
    # Create cumsum x
    x_center_mean = x[border:-border, border:-border].mean()
    x_csum = (np.float32(x)-x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    # Save images
    X_test[n, ..., 0] = x.squeeze() / 255
    X_test[n, ..., 1] = x_csum.squeeze()

print('Done!')
# Normalize X_test_feats
X_feat_test -= x_feat_mean
X_feat_test /= x_feat_std
# Load best model
model.load_weights('model-tgs-salt-1.h5')
# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate({'img': X_valid, 'feat': X_feat_valid}, y_valid, verbose=1)
# Predict on train, val and test
preds_train = model.predict({'img': X_train, 'feat': X_feat_train}, verbose=1)
preds_val = model.predict({'img': X_valid, 'feat': X_feat_valid}, verbose=1)
preds_test = model.predict({'img': X_test, 'feat': X_feat_test}, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
# Create list of upsampled test masks
preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))
preds_test_upsampled[0].shape
def plot_sample(X, y, preds):
    ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(X[ix, ..., 1], cmap='seismic')
    if has_mask:
        ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('Seismic cumsum')

    ax[2].imshow(y[ix].squeeze())
    ax[2].set_title('Salt')

    ax[3].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Salt Pred');
# src: https://www.kaggle.com/aglotero/another-iou-metric
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
thres = np.linspace(0.25, 0.75, 20)
thres_ioc = [iou_metric_batch(y_valid, np.int32(preds_val > t)) for t in tqdm_notebook(thres)]
plt.plot(thres, thres_ioc);
best_thres = thres[np.argmax(thres_ioc)]
best_thres, max(thres_ioc)
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

pred_dict = {id_[:-4]:RLenc(np.round(preds_test_upsampled[i] > best_thres)) for i,id_ in tqdm_notebook(enumerate(test_ids))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
sub.head()
