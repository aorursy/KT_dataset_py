import os
train_img_dir = '../input/train/images/'
train_mask_dir = '../input/train/masks/'
test_img_dir = '../input/test/images/'
train_img_names = [x.split('.')[0] for x in os.listdir(train_img_dir)]
train_img_names[:5],len(train_img_names)
train_img_dict_i_to_names = dict()
train_img_dict_names_to_i = dict()
for i in range(len(train_img_names)):
    train_img_dict_i_to_names[i] = train_img_names[i]
    train_img_dict_names_to_i[train_img_names[i]] = i
from skimage.data import imread
train_img_shape = imread(train_img_dir + train_img_names[0]+'.png').shape
train_mask_shape = imread(train_mask_dir + train_img_names[0]+'.png').shape
import numpy as np
train_img = np.zeros((len(train_img_names), train_img_shape[0], train_img_shape[1], train_img_shape[2]))
train_mask = np.zeros((len(train_img_names), train_mask_shape[0], train_mask_shape[1]))
for i in range(len(train_img_names)):
    train_img[i] = i
    train_mask[i] = i
    train_img[i,:,:,:] = imread(train_img_dir + train_img_names[i]+'.png')
    train_mask[i,:,:] = imread(train_mask_dir + train_img_names[i]+'.png')
train_img.shape,train_mask.shape
train_img[50,:,:,0],train_mask[50,:,:]
train_img_mono = np.zeros((len(train_img_names), train_img_shape[0], train_img_shape[1]))
train_img_mono = train_img[:,:,:,0]
train_img_mono.shape
train_mask_8bit = np.zeros((train_mask.shape[0],train_mask.shape[1],train_mask.shape[1]))
for i in range(len(train_img_names)):
    train_mask_8bit[i,:,:]= np.maximum(train_mask[i,:,:]/255-2,0)
train_mask_8bit[50,:,:]
import pandas as pd
train_dir = '../input/'
train = pd.read_csv(train_dir + 'train.csv')
train.head(3)
train.shape
depths = pd.read_csv(train_dir + 'depths.csv')
depths.head(3)
depths.shape
train = pd.merge(train, depths, on='id',how='left')
train.head(3)
train.shape
def rle_to_mask(rle_list, SHAPE):
    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask
def calc_area_for_rle(rle_str):
    rle_list = str(rle_str).split()
    mask = rle_to_mask(rle_list, (101,101))
    area = mask.sum()/255.0
    return area
train['area'] = train['rle_mask'].apply(calc_area_for_rle)
train.head(3)
def calc_mean_img(name):
    i = train_img_dict_names_to_i[name]
    img = train_img_mono[i]
    mean = img.mean()
    return mean
train['mean'] = train['id'].apply(calc_mean_img)
train.head(3)
def calc_std_img(name):
    i = train_img_dict_names_to_i[name]
    img = train_img_mono[i]
    std = img.std()
    return std
train['std'] = train['id'].apply(calc_std_img)
train.head(3)
train_issalt = train[train['rle_mask'].notnull()]
train_nosalt = train[train['rle_mask'].isnull()]
train.shape,train_issalt.shape,train_nosalt.shape
train_nosalt.shape[0]/train.shape[0]
import matplotlib.pyplot as plt
%matplotlib inline
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].hist(train['area'], bins=20)
axes[0].set_title('train')
axes[0].set_xlabel('area')
axes[0].set_ylabel('frequency')
axes[1].hist(train_issalt['area'], bins=20)
axes[1].set_title('train_issalt')
axes[1].set_xlabel('area')
axes[1].set_ylabel('frequency')
axes[2].hist(train_nosalt['area'], bins=20)
axes[2].set_title('train_nosalt')
axes[2].set_xlabel('area')
axes[2].set_ylabel('frequency')
small_area_image_list = train_issalt[train_issalt['area'] < 101*101*0.01]['id'].tolist()
small_area_image_list[:5],len(small_area_image_list)
image_list = small_area_image_list[:5]
fig, axes = plt.subplots(len(image_list), 2, figsize=(5,5*len(image_list)))
fig.subplots_adjust(left=0.075,right=0.95,bottom=0.05,top=0.52,wspace=0.2,hspace=0.10)
for i in range(len(image_list)):
    img = imread(train_img_dir + image_list[i] +'.png')
    mask = imread(train_mask_dir + image_list[i] +'.png')
    axes[i, 0].imshow(img)
    axes[i, 1].imshow(mask)
large_area_image_list = train_issalt[train_issalt['area'] > 101*101*0.99]['id'].tolist()
large_area_image_list[:5],len(large_area_image_list)
image_list = large_area_image_list[:5]
fig, axes = plt.subplots(len(image_list), 2, figsize=(5,5*len(image_list)))
fig.subplots_adjust(left=0.075,right=0.95,bottom=0.05,top=0.52,wspace=0.2,hspace=0.10)
for i in range(len(image_list)):
    img = imread(train_img_dir + image_list[i] +'.png')
    mask = imread(train_mask_dir + image_list[i] +'.png')
    axes[i, 0].imshow(img)
    axes[i, 1].imshow(mask)
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].hist(train['z'], bins=20)
axes[0].set_title('train')
axes[0].set_xlabel('depth')
axes[0].set_ylabel('frequency')
axes[0].set_ylim(0,350)
axes[1].hist(train_issalt['z'], bins=20)
axes[1].set_title('train_issalt')
axes[1].set_xlabel('depth')
axes[1].set_ylabel('frequency')
axes[1].set_ylim(0,350)
axes[2].hist(train_nosalt['z'], bins=20)
axes[2].set_title('train_nosalt')
axes[2].set_xlabel('depth')
axes[2].set_ylabel('frequency')
axes[2].set_ylim(0,350)
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].hist(train['mean'], bins=20)
axes[0].set_title('train')
axes[0].set_xlabel('brightness mean')
axes[0].set_ylabel('frequency')
axes[0].set_ylim(0,1000)
axes[1].hist(train_issalt['mean'], bins=20)
axes[1].set_title('train_issalt')
axes[1].set_xlabel('brightness mean')
axes[1].set_ylabel('frequency')
axes[1].set_ylim(0,1000)
axes[2].hist(train_nosalt['mean'], bins=20)
axes[2].set_title('train_nosalt')
axes[2].set_xlabel('brightness mean')
axes[2].set_ylabel('frequency')
axes[2].set_ylim(0,1000)
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].hist(train['std'], bins=20)
axes[0].set_title('train')
axes[0].set_xlabel('brightness std')
axes[0].set_ylabel('frequency')
axes[0].set_ylim(0,500)
axes[1].hist(train_issalt['std'], bins=20)
axes[1].set_title('train_issalt')
axes[1].set_xlabel('brightness std')
axes[1].set_ylabel('frequency')
axes[1].set_ylim(0,500)
axes[2].hist(train_nosalt['std'], bins=20)
axes[2].set_title('train_nosalt')
axes[2].set_xlabel('brightness std')
axes[2].set_ylabel('frequency')
axes[2].set_ylim(0,500)
xlabel = 'depth'
ylabel = 'area'
x = 'z'
y = 'area'
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].scatter(train[x], train[y])
axes[0].set_title('train')
axes[0].set_xlabel(xlabel)
axes[0].set_ylabel(ylabel)
axes[1].scatter(train_issalt[x], train_issalt[y])
axes[1].set_title('train_issalt')
axes[1].set_xlabel(xlabel)
axes[1].set_ylabel(ylabel)
axes[2].scatter(train_nosalt[x], train_nosalt[y])
axes[2].set_title('train_nosalt')
axes[2].set_xlabel(xlabel)
axes[2].set_ylabel(ylabel)
xlabel = 'depth'
ylabel = 'brightness mean'
x = 'z'
y = 'mean'
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].scatter(train[x], train[y])
axes[0].set_title('train')
axes[0].set_xlabel(xlabel)
axes[0].set_ylabel(ylabel)
axes[0].set_ylim(0,260)
axes[1].scatter(train_issalt[x], train_issalt[y])
axes[1].set_title('train_issalt')
axes[1].set_xlabel(xlabel)
axes[1].set_ylabel(ylabel)
axes[1].set_ylim(0,260)
axes[2].scatter(train_nosalt[x], train_nosalt[y])
axes[2].set_title('train_nosalt')
axes[2].set_xlabel(xlabel)
axes[2].set_ylabel(ylabel)
axes[2].set_ylim(0,260)
xlabel = 'depth'
ylabel = 'brightness std'
x = 'z'
y = 'std'
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].scatter(train[x], train[y])
axes[0].set_title('train')
axes[0].set_xlabel(xlabel)
axes[0].set_ylabel(ylabel)
axes[0].set_ylim(0,80)
axes[1].scatter(train_issalt[x], train_issalt[y])
axes[1].set_title('train_issalt')
axes[1].set_xlabel(xlabel)
axes[1].set_ylabel(ylabel)
axes[1].set_ylim(0,80)
axes[2].scatter(train_nosalt[x], train_nosalt[y])
axes[2].set_title('train_nosalt')
axes[2].set_xlabel(xlabel)
axes[2].set_ylabel(ylabel)
axes[2].set_ylim(0,80)
xlabel = 'area'
ylabel = 'brightness mean'
x = 'area'
y = 'mean'
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].scatter(train[x], train[y])
axes[0].set_title('train')
axes[0].set_xlabel(xlabel)
axes[0].set_ylabel(ylabel)
axes[0].set_ylim(0,300)
axes[1].scatter(train_issalt[x], train_issalt[y])
axes[1].set_title('train_issalt')
axes[1].set_xlabel(xlabel)
axes[1].set_ylabel(ylabel)
axes[1].set_ylim(0,300)
axes[2].scatter(train_nosalt[x], train_nosalt[y])
axes[2].set_title('train_nosalt')
axes[2].set_xlabel(xlabel)
axes[2].set_ylabel(ylabel)
axes[2].set_ylim(0,300)
xlabel = 'area'
ylabel = 'brightness std'
x = 'area'
y = 'std'
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].scatter(train[x], train[y])
axes[0].set_title('train')
axes[0].set_xlabel(xlabel)
axes[0].set_ylabel(ylabel)
axes[0].set_ylim(0,80)
axes[1].scatter(train_issalt[x], train_issalt[y])
axes[1].set_title('train_issalt')
axes[1].set_xlabel(xlabel)
axes[1].set_ylabel(ylabel)
axes[1].set_ylim(0,80)
axes[2].scatter(train_nosalt[x], train_nosalt[y])
axes[2].set_title('train_nosalt')
axes[2].set_xlabel(xlabel)
axes[2].set_ylabel(ylabel)
axes[2].set_ylim(0,80)
xlabel = 'brightness mean'
ylabel = 'brightness std'
x = 'mean'
y = 'std'
fig, axes = plt.subplots(1, 3, figsize=(7*3,5))
axes[0].scatter(train[x], train[y])
axes[0].set_title('train')
axes[0].set_xlabel(xlabel)
axes[0].set_ylabel(ylabel)
axes[0].set_ylim(0,80)
axes[1].scatter(train_issalt[x], train_issalt[y])
axes[1].set_title('train_issalt')
axes[1].set_xlabel(xlabel)
axes[1].set_ylabel(ylabel)
axes[1].set_ylim(0,80)
axes[2].scatter(train_nosalt[x], train_nosalt[y])
axes[2].set_title('train_nosalt')
axes[2].set_xlabel(xlabel)
axes[2].set_ylabel(ylabel)
axes[2].set_ylim(0,80)
def suspicious_img_c(ids):
    mask = imread(train_mask_dir + ids +'.png')
    if len(np.unique(mask.sum(axis=1)))==1:
        if mask.sum() == 101*101*65535:
            return 0
        elif mask.sum() == 0:
            return 0
        else:
            return 1
    else:
        return 0
def suspicious_img_r(ids):
    mask = imread(train_mask_dir + ids +'.png')
    if len(np.unique(mask.sum(axis=0)))==1:
        if mask.sum() == 101*101*65535:
            return 0
        elif mask.sum() == 0:
            return 0
        else:
            return 1
    else:
        return 0
train['suspicious_c'] = train['id'].map(suspicious_img_c)
train['suspicious_r'] = train['id'].map(suspicious_img_r)
train_suspicious_c = train[train['suspicious_c']==1]
train_suspicious_r = train[train['suspicious_r']==1]
train_suspicious_c.shape[0], train_suspicious_r.shape[0]
train_suspicious_list = train_suspicious_c['id'].tolist()
image_list = train_suspicious_list[:30]
fig, axes = plt.subplots(len(image_list), 2, figsize=(5,5*len(image_list)))
fig.subplots_adjust(left=0.075,right=0.95,bottom=0.05,top=0.52,wspace=0.2,hspace=0.10)
for i in range(len(image_list)):
    img = imread(train_img_dir + image_list[i] +'.png')
    mask = imread(train_mask_dir + image_list[i] +'.png')
    axes[i, 0].imshow(img)
    axes[i, 1].imshow(mask)