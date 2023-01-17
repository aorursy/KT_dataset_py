%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import pickle
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()
# from tqdm import tqdm

# pd.options.display.max_rows = 999
# pd.options.display.max_columns = 999
import glob
def get_path(str, first=True, parent_dir='../input/**/'):
    res_li = glob.glob(parent_dir+str)
    return res_li[0] if first else res_li
DATA_DIR = '../input/dogs-vs-cats-redux-kernels-edition/'
evals = pd.read_csv('../input/dvc-prepare-evalset/evals.csv')
evals.head()
H, W, C = 128, 128, 3
import cv2
tmp = cv2.cvtColor(
    cv2.imread(DATA_DIR+'train/cat.0.jpg', cv2.IMREAD_COLOR), 
    cv2.COLOR_BGR2RGB
).astype('float')/255.
print('shape', tmp.shape)
tmp_rsz = cv2.resize(tmp, (H, W), interpolation=cv2.INTER_NEAREST)
print('resized', tmp_rsz.shape)
plt.figure(figsize=[12, 8])
plt.subplot(1, 2, 1)
plt.imshow(tmp); plt.title('Original'); 
plt.subplot(1, 2, 2)
plt.imshow(tmp_rsz); plt.title('Resized'); 
from skimage import exposure
tmp = tmp_rsz.copy()
tmp_bri_norm = (tmp - np.mean(tmp))/np.std(tmp)
tmp_bri_norm -= tmp_bri_norm.min()
tmp_bri_norm /= tmp_bri_norm.max()
tmp_hist_eq = exposure.equalize_adapthist(tmp.copy(), clip_limit=0.01)

def plot_hist(img):
    for i in range(3):
        sns.distplot(img[:, :, i].ravel());
        plt.legend(['R', 'G', 'B']);

plt.figure(figsize=[22, 18])
plt.subplot(1, 4, 1)
plt.imshow(tmp); plt.title('Resized'); 
plt.subplot(1, 4, 2)
plt.imshow(tmp_bri_norm); plt.title('Bright_normalized'); 
plt.subplot(1, 4, 3)
plt.imshow(tmp_hist_eq); plt.title('Histogram_equalized limit 0.01'); 
plt.subplot(1, 4, 4)
clip_limit = 0.1
plt.imshow(exposure.equalize_adapthist(tmp, clip_limit=clip_limit)); 
plt.title(f'Histogram_equalized limit {clip_limit}'); 

plt.figure(figsize=[22, 4])
plt.subplot(1, 4, 1)
plot_hist(tmp); plt.title('Resized'+' Hist'); 
plt.subplot(1, 4, 2)
plot_hist(tmp_bri_norm); plt.title('Bright_normalized'+' Hist'); 
plt.subplot(1, 4, 3)
plot_hist(tmp_hist_eq); plt.title(f'Histogram_equalized limit 0.01'+' Hist'); 
plt.subplot(1, 4, 4)
plot_hist(exposure.equalize_adapthist(tmp, clip_limit=0.03)); 
plt.title(f'Histogram_equalized limit {clip_limit}'+' Hist'); 
import tensorflow as tf
import keras
from keras.preprocessing import image
imgGen = image.ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
)
imgGen_test = image.ImageDataGenerator(rescale=1./255,)
train_gen = imgGen.flow_from_directory(
    DATA_DIR+'train/',
    class_mode=None, 
    target_size=(H, W),
    batch_size=32,
)

valid_fold = 0
train_gen.class_indices = {'dog': 0, 'cat': 1}
mask = (evals['is_test']==0) & (evals['eval_set']!=valid_fold)
train_gen.filenames = evals.loc[mask, 'img_id'].apply(lambda x: x+'.jpg').values.tolist()
train_gen.classes = evals.loc[mask, 'target'].values
train_gen.class_mode = 'binary'
train_gen.samples = len(evals.loc[mask, 'target'].values)
train_gen.n = len(evals.loc[mask, 'target'].values)
train_gen.num_classes = 2

for bx, by in train_gen:
    break
print('targets(is_cat?)', [by[i] for i in range(12)])
plt.figure(figsize=[12, 8])
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(bx[i]);
    plt.axis('off');
test_gen = imgGen_test.flow_from_directory(
    DATA_DIR+'test/',
    class_mode=None, 
    target_size=(H, W),
    batch_size=32,
)

valid_fold = 0
# test_gen.class_indices = {'dog': 0, 'cat': 1}
mask = (evals['is_test']==1)
test_gen.filenames = evals.loc[mask, 'img_id'].apply(lambda x: x+'.jpg').values.tolist()
test_gen.classes = evals.loc[mask, 'target'].values
test_gen.class_mode = 'binary'
test_gen.samples = len(evals.loc[mask, 'target'].values)
test_gen.n = len(evals.loc[mask, 'target'].values)
test_gen.num_classes = 2

for bx, by in test_gen:
    break
print('targets(is_cat?)', [by[i] for i in range(12)])
plt.figure(figsize=[12, 8])
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(bx[i]);
    plt.axis('off');



