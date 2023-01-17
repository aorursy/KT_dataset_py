# Set Commit/Development mode
COMMIT = True
DEV = not COMMIT
DBG = False
def mem_used():
    """Memory used"""
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF)[2] * 10/1028 / 10, 1)

def mem_fun(fun, **kwargs):
    """"""
    mem_start = mem_used()
    _ = fun(**kwargs)
    print(f'memory used by function: {(mem_used() - mem_start):.1f}mb')

nb_mem = mem_used()
f'Initial memory used by this notebook: {nb_mem}mb'
import os
import gc
import sys
import random
from memory_profiler import profile

import pandas as pd
import numpy as np
from itertools import chain

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
%matplotlib inline

from tqdm import tqdm_notebook

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize

from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Dropout, BatchNormalization, UpSampling2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import backend as K
path = '../input'
path_train = f'{path}/train'
path_test = f'{path}/test'
imgs_train = f'{path}/train/images'
masks_train = f'{path}/train/masks'
imgs_test = f'{path}/test/images'
IMG_SIZE = 101   # original/raw image size
TGT_SIZE = 128   # model/input image size
# Check memory allocation needed for train/test set
def test_alloc_size(n, m, c):
    """"""
    imgs_test_alloc = list(range(n))
    for i in range(n):
        imgs_test_alloc[i] = np.ones((m, TGT_SIZE, TGT_SIZE, c)) * random.randint(0,100)
    return imgs_test_alloc
    
if DBG:
    mem_fun(test_alloc_size, n=1, m=18000, c=2)
    print(f'notebook memory used: {mem_used()}mb')
    ## memory used by function: 4482.3mb for (1800, 128, 1288, 2)
    ## notebook memory used: 4805.7mb
print(f'notebook memory used: {mem_used()}mb')
# Helper functions
def upsample(img, img_size_target=TGT_SIZE):
    """Resize image to target shape(model_input) or back to original shape"""
    if img.shape[0] == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def imgs_2_array(path, img_names, ftype='.png', size=TGT_SIZE, flip=True):
    """Load images and transform to array with image and cumsum layer"""
    imgs = np.zeros((len(img_names) * (1 + flip), TGT_SIZE, TGT_SIZE, 1))
    imgs[:len(img_names), ..., :1] = np.array([upsample(np.array(load_img(f'{path}/{name}{ftype}', grayscale=True))) / 255
                      for name in tqdm_notebook(img_names)]).reshape(-1, TGT_SIZE, TGT_SIZE, 1)
    if flip:
        print('...extend set with flipped images')
        imgs[len(img_names):, ..., :1] = np.array([np.fliplr(img) 
                                for img in imgs[:len(img_names), ..., :1]]).reshape(-1, TGT_SIZE, TGT_SIZE, 1)
    return imgs
        
    
def csum(img, weight=.5, border=5):
    """Create image cumsum from image
    Sort of image bleeding downwards"""
    center_mean = img[border:-border, border:-border].mean()
    csum = (np.float32(img)-center_mean).cumsum(axis=0)         
    csum -= csum[border:-border, border:-border].mean()
    csum /= max(1e-3, csum[border:-border, border:-border].std())
    return csum * weight

def clip_norm(img, weight=1.96):
    """Normalized and clipped image for second image layer"""
    img = np.clip(img, -weight*img.std(), weight*img.std())
    return (img - img.mean()) / img.std()

def imgs_2_fn(path, img_names, ftype='.png', size=TGT_SIZE, flip=True, weight=1, fn=clip_norm):
    """Load images and transform to array with image and cumsum layer"""
    imgs = np.zeros((len(img_names) * (1 + flip), TGT_SIZE, TGT_SIZE, 2))
    imgs[:len(img_names), ..., :1] = np.array([upsample(np.array(load_img(f'{path}/{name}{ftype}', grayscale=True))) / 255
                      for name in tqdm_notebook(img_names)]).reshape(-1, TGT_SIZE, TGT_SIZE, 1)
    if flip:
        print('...extend set with flipped images')
        imgs[len(img_names):, ..., :1] = np.array([np.fliplr(img) 
                                for img in imgs[:len(img_names), ..., :1]]).reshape(-1, TGT_SIZE, TGT_SIZE, 1)
    imgs[..., 1] = [fn(img, weight) for img in tqdm_notebook(imgs[..., 0])]
    return imgs
train_df_ = pd.read_csv(f'{path}/train.csv', index_col="id", usecols=[0])
depths_df_ = pd.read_csv(f'{path}/depths.csv', index_col="id") # train and test
train_df_ = train_df_.join(depths_df_)
test_df = depths_df_[~depths_df_.index.isin(train_df_.index.values)]

# Indices
train_indices = train_df_.index.values
test_indices = test_df.index.values
# Free up some RAM
del depths_df_
print(f'notebook memory used: {mem_used()}mb')
# Flip(augment) train images -> first duplicate train_df: images & depth
train_df = pd.concat([train_df_, train_df_])
train_df.index = np.concatenate([train_indices, train_indices+'_'])
if DEV:
    print(train_df.index[:5], train_df.index[4000:4005])
print('Loading train set images...')
# Use without second layer
X_imgs = imgs_2_array(imgs_train, train_indices, '.png', TGT_SIZE)

# Use with second layer
# TODO: Train & test results are less with this second layer
# X_imgs = imgs_2_fn(imgs_train, train_indices, '.png', TGT_SIZE, weight=0)

print('Loading train set masks...')
X_masks = imgs_2_array(masks_train, train_indices, '.png', TGT_SIZE)

print('Computing salt mask coverage...')
X_coverages = np.array([np.sum(mask) / (mask.shape[0]*mask.shape[1]) for mask in X_masks])
X_cov_class = (X_coverages - .01) * 100//10 + 1

# Normalize depth
print('Computing normalized seismic dept...')
depth = train_df["z"]
mean_depth, std_depth, max_depth = depth.mean(), depth.std(), depth.max()
X_norm_depth = (depth - mean_depth) / std_depth

print(f'Loading ready.\nnotebook memory used: {mem_used()}mb')
# Sanity check classes are correct
if DEV:
    X_coverages[:10], X_cov_class[:10]
# Sanity check flip image have same depth
if DEV:
    X_norm_depth[:5].values == X_norm_depth[4000:4005].values
if DEV:
    _ = sns.distplot(train_df.z, label="Train")
    _ = sns.distplot(test_df.z, label="Test")
    _ = plt.legend()
    _ = plt.title("Depth distribution")
# Helper functions for printing masks
def coverage(mask):
    """Compute salt mask coverage"""
    return np.sum(mask) / (mask.shape[0]*mask.shape[1])


def norm_coverage(masks):
    """Compute salt mask coverage"""
    coverages = np.array([coverage(mask) for mask in masks])
    mean_cov, std_cov, max_cov = coverages.mean(), coverages.std(), coverages.max()
    return (coverages - mean_cov) / std_cov


def coverage_class(mask):
    """Compute salt mask coverage class"""
    if coverage(mask) == 0:
        return 0
    return (coverage(mask) * 100 //10).astype(np.int8) +1
if DEV:
    _ = sns.distplot(X_cov_class, label="Train", kde=False)
    _ = plt.legend()
    _ = plt.title("Coverage distribution")
if DEV:
    salt_cover_norm = (X_coverages - np.mean(X_coverages)) / np.std(X_coverages)

    plt.figure(figsize=(20,10))
    plt.scatter(range(len(X_norm_depth)), X_norm_depth, alpha=.5, label='Normalized Seismic Depth')
    plt.scatter(range(len(salt_cover_norm)), salt_cover_norm, color='r', alpha=.5, label='Normalized Salt Coverage')
    plt.title('Normalized Depth vs. Salt coverage as % of image size', fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, fontsize=16);
    plt.show();
def plot_imgs_masks(imgs, masks, **kwargs):
    """Visualize seismic images with their salt area mask(green) and optionally salt area prediction(pink). 
    The prediction mask can be either in probability-mask or binary-mask form(based on threshold)
    """
    depth = kwargs.get('depth', None)
    preds_valid = kwargs.get('preds_valid', None)
    thres = kwargs.get('thres', None)
    grid_width = kwargs.get('grid_width', 10)
    zoom = kwargs.get('zoom', 1.5)
    
    grid_height = 1 + (len(imgs)-1) // grid_width
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width*zoom, grid_height*zoom))
    axes = axs.ravel()
    
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        
        ax = axes[i] #//grid_width, i%grid_width]
        _ = ax.imshow(img[..., 0], cmap="Greys")
#         _ = ax.imshow(img[..., 1], alpha=0.15, cmap="seismic") # TODO
        _ = ax.imshow(mask[..., 0], alpha=0.3, cmap="Greens")
        
        if preds_valid is not None:
            pred = preds_valid[i]
            pred = pred[..., 0]
            if thres is not None:
                pred = np.array(np.round(pred > thres), dtype=np.float32)
                iou = f'IoU: {_iou(mask, pred).round(3)}'
                _ = ax.imshow(pred, alpha=0.3, cmap="OrRd")
                _ = ax.text(2, img.shape[0]-2, iou, color="k")
            else:
                _ = ax.imshow(pred, alpha=0.3, cmap="OrRd")
            
        if depth is not None:
            _ = ax.text(2, img.shape[0]-2, f'depth: {depth[i]}', color="k")
            
        _ = ax.text(2, 2, f'{coverage(mask).round(3)}({coverage_class(mask)})', color="k", ha="left", va="top")
        _ = ax.set_yticklabels([])
        _ = ax.set_xticklabels([])
        _ = plt.axis('off')
    plt.suptitle("Green: Salt area mask \nTop-left: coverage class, top-right: salt coverage, bottom-left: depth", y=1+.5/grid_height, fontsize=20)
    plt.tight_layout();
if DEV:
    N = 30
    plot_imgs_masks(X_imgs[:N], X_masks[:N], depth=train_df.iloc[:N].z)
VAL_SIZE = 0.20

print(f'notebook memory used before split: {mem_used()}mb')

X_train, X_valid, Y_train, Y_valid, depth_train, depth_valid = train_test_split(
    X_imgs,
    X_masks,
    np.array(X_norm_depth).reshape(-1, 1),
    test_size=VAL_SIZE, 
    stratify=X_cov_class, 
    random_state=1)

gc.collect()
del X_imgs, X_masks, X_cov_class
gc.collect()
print(f'notebook memory used after split: {mem_used()}mb')
if DEV:
    print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, depth_train.shape, depth_valid.shape)
if DEV:
    N = 20
    plot_imgs_masks(X_valid[:N], Y_valid[:N])
def mean_iou(Y_true, Y_pred, score_thres=0.5):
    """Compute mean(IoU) metric
    IoU = intersection / union
    
    For each (mask)threshold in provided range:
     - convert probability mask to boolean mask based on given threshold
     - score the mask 1 if(IoU > score_threshold(0.5))
    Take the mean of the scoress

    https://www.tensorflow.org/api_docs/python/tf/metrics/mean_iou
    """
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        Y_pred_bool = tf.to_int32(Y_pred > t) # boolean mask by threshold
        score, update_op = tf.metrics.mean_iou(Y_true, Y_pred_bool, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            score = tf.identity(score) #!! use identity to transform score to tensor
        prec.append(score) 
        
    return K.mean(K.stack(prec), axis=0)
def conv_block(m, ch_dim, acti, bn, res, do=0):
    """CNN block"""
    n = Conv2D(ch_dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(ch_dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def input_feature(f, n, n_features=1):
    """Input block"""
    features = 1
    xx = K.int_shape(n)[1]
    f_repeat = RepeatVector(xx*xx)(f)
    f_conv = Reshape((xx, xx, n_features))(f_repeat)
    n = Concatenate(axis=-1, name=f'feat_{2}')([n, f_conv])
    n = BatchNormalization()(n)            
    return n
def level_block(m, ch_dim, depth, inc_rate, acti, do, bn, mp, up, res, inp_feat):
    """Recursive CNN builder"""
    if depth > 0:
        n = conv_block(m, ch_dim, acti, bn, res) # no drop-out
        m = MaxPooling2D()(n) if mp else Conv2D(ch_dim, 3, strides=2, padding='same')(n)
        if (inp_feat is not None) and (depth==2):
            m = Concatenate()([m, input_feature(inp_feat, m)])
        m = level_block(m, int(inc_rate*ch_dim), depth-1, inc_rate, acti, do, bn, mp, up, res, inp_feat)
        
        # Unwind recursive stack calls - creating the upscaling part of the model
        if up:
            # Repeat the rows and columns of the data by 2 and 2 respectively
            m = UpSampling2D()(m)
            m = Conv2D(ch_dim, 2, activation=acti, padding='same')(m)
        else:
            # Transposed convolutions are going in the opposite direction of a normal convolution
            m = Conv2DTranspose(ch_dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, ch_dim, acti, bn, res)
    else:
        # Depth == 0 - deepest conv_block
        m = conv_block(m, ch_dim, acti, bn, res, do)
    return m
def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
        dropout=0.5, batchnorm=False, maxpool=True, upconv=False, residual=False):
    """Returns model"""
    inputs = Input(shape=img_shape, name='img')
    inp_feat = Input(shape=(1,), name='feat') # or None
    outputs = level_block(inputs, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual, inp_feat)
    outputs = Conv2D(out_ch, 1, activation='sigmoid')(outputs)
    return Model(inputs=[inputs, inp_feat], outputs=outputs)
IMG_CH = 1     # layers of image
CONV_CH = 16   # number of channels to start/end UNet with
DEPTH = 5      # number of CONV blocks to max model depth
D_OUT = 0.1
BN = True
UP_CONV = False
RES = True

model = UNet((TGT_SIZE, TGT_SIZE, IMG_CH), 
             start_ch=CONV_CH, 
             depth=DEPTH, 
             dropout=D_OUT,
             batchnorm=BN, 
             upconv=UP_CONV,
             residual=RES)
LR = 10e-3
# Define optimizer
# Clip gradients to norm 1., 
optimizer = [Adam(lr=LR, beta_1=0.9, beta_2=0.9999, decay=LR/100, clipvalue=.5),
             SGD(lr=LR, decay=LR/100, momentum=0.9, nesterov=True, clipnorm=1.)]

# Define loss
loss = ["binary_crossentropy", "kullback_leibler_divergence"]

# Compile model
model.compile(loss=loss[0], optimizer=optimizer[0], metrics=["accuracy", mean_iou])
model_name = f'TGS_salt_UNet_{IMG_CH}_{CONV_CH}_{DEPTH}_{D_OUT>0}_{BN}_{UP_CONV}_{RES}.h5'
model.summary()
BATCH_SIZE = 32 # larger will have more stable learning, but needs more GPU
EPOCHS = 50
if DEV:
    EPOCHS = int(input('Number of training epochs?:'))
callbacks = [
    EarlyStopping(patience=11, verbose=1),
    ReduceLROnPlateau(patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, verbose=1)]

X_train_dict = {'img': X_train, 
              'feat': depth_train}

X_val_dict = {'img': X_valid, 
              'feat': depth_valid}

history = model.fit(X_train_dict, 
                    Y_train, 
                    validation_data=(X_val_dict, Y_valid),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=callbacks)
if DEV:
    print(history.history.keys())
if DEV:
    fig, (ax_loss, ax_acc, ax_iou) = plt.subplots(1, 3, figsize=(15,5))

    _ = ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    _ = ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    _ = ax_loss.legend()
    _ = ax_loss.set_title('Loss')
    
    _ = ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
    _ = ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
    _ = ax_acc.legend()
    _ = ax_acc.set_title('Accuracy')
    
    _ = ax_iou.plot(history.epoch, history.history["mean_iou"], label="Train IoU")
    _ = ax_iou.plot(history.epoch, history.history["val_mean_iou"], label="Validation IoU")
    _ = ax_iou.legend()
    _ = ax_iou.set_title('IoU')
# Load best model
model = load_model(model_name, custom_objects={'mean_iou': mean_iou})
print('model loading done.')
# Evaluate best model on validation set
if DEV:
    model.evaluate(X_val_dict, Y_valid, verbose=1)
# Predict on validation set
preds_valid = model.predict(X_val_dict, verbose=1).reshape(-1, TGT_SIZE, TGT_SIZE)
preds_valid = preds_valid.reshape(-1, TGT_SIZE, TGT_SIZE, 1)
if DEV:
    print(preds_valid.shape, Y_valid.shape)
if DEV:
    N = 40
    plot_imgs_masks(X_valid[:N], Y_valid[:N], preds_valid=preds_valid[:N])
def _iou(Y_true, Y_pred):
    """IoU"""
    Y_true_f, Y_pred_f = Y_true.ravel(), Y_pred.ravel()
    intersection = np.sum(Y_true_f * Y_pred_f)
    union = np.sum((Y_true_f + Y_pred_f) > 0)
    return intersection/ max(1e-9, union)

def miou(Y_trues, Y_preds):
    """Mean intersection over union"""
    return np.mean([_iou(Y_trues[i], Y_preds[i]) for i in range(Y_trues.shape[0])])
thresholds = np.linspace(0.1, 0.9, 80)
ious = np.array([miou(Y_valid, np.int32(preds_valid > threshold)) 
                 for threshold in tqdm_notebook(thresholds)])
threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]
if DEV:
    _ = plt.plot(thresholds, ious)
    _ = plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    _ = plt.xlabel("Threshold")
    _ = plt.ylabel("IoU")
    _ = plt.title("Threshold: {} delivers best mean-IoU: {} ".format(threshold_best.round(3), iou_best.round(3)))
    _ = plt.legend()
if DEV:
    N = 60
    plot_imgs_masks(X_valid[:N], Y_valid[:N], preds_valid=preds_valid[:N], thres=threshold_best)
# Free up memory: need ~7gb for test set
gc.collect()
print(f'notebook memory used: {mem_used()}mb')


if DEV:
    test_ids = next(os.walk(path_test+"/images"))[2]
    assert len(set(test_ids) ^ set(test_df.index+'.png')) == 0
# Load test set (keep >5gb free RAM!)
X_test = imgs_2_array(imgs_test, test_indices, '.png', TGT_SIZE, flip=False)
gc.collect()
print(f'notebook memory used: {mem_used()}mb')
# Normalize depth - !use train mean and std
depth_test = (test_df["z"] - mean_depth) / std_depth
if DEV:
    print(X_test.shape)
X_test_dict = {'img': X_test, 
              'feat': depth_test} 

preds_test = model.predict(X_test_dict) 
def RLenc(img, order='F'):
    """Convert binary mask image to run-length array or string.
    
    Args:
    img: image in shape [n, m]
    order: is down-then-right, i.e. Fortran(F)
    string: return in string or array

    Return:
    run-length as a string: <start[1s] length[1s] ... ...>
    """
    bytez = img.reshape(img.shape[0] * img.shape[1], order=order)
    bytez = np.concatenate([[0], bytez, [0]])
    runs = np.where(bytez[1:] != bytez[:-1])[0] + 1 # pos start at 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# Use for sanity check the encode function
def RLdec(rl_string, shape=(101, 101), order='F'):
    """Convert run-length string to binary mask image.
    
    Args:
    rl_string: 
    shape: target shape of array
    order: decode order is down-then-right, i.e. Fortran(F)

    Return:
    binary mask image as array
    """
    s = rl_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order=order)
pred_dict = {idx: RLenc(np.round(upsample(preds_test[i], IMG_SIZE) > threshold_best)) 
             for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
sub.head()
print('submission saved!')