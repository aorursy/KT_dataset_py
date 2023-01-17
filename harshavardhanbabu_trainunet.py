# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import zipfile

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

import cv2

import fastai

import torch

from matplotlib import pyplot as plt

from pathlib import Path

import matplotlib

import shutil

from tqdm import tqdm_notebook as tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/"))

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import fastai



def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):

    if not tfms: tfms=(None,None)

    assert is_listy(tfms) and len(tfms) == 2

    self.train.transform(tfms[0], **kwargs)

    self.valid.transform(tfms[1], **kwargs)

    kwargs['tfm_y'] = False # Test data has no labels

    if self.test: self.test.transform(tfms[1], **kwargs)

    return self



fastai.data_block.ItemLists.transform = transform
def dice(input:Tensor, targs:Tensor, eps:float=1e-8)->Rank0Tensor:

    input = input.clone()

    targs = targs.clone()

    n = targs.shape[0]

    input = torch.softmax(input, dim=1).argmax(dim=1)

    input = input.view(n, -1)

    targs = targs.view(n, -1)

    input[input == 0] = -999

    intersect = (input == targs).sum().float()

    union = input[input > 0].sum().float() + targs[targs > 0].sum().float()

    del input, targs

    gc.collect()

    return ((2.0 * intersect + eps) / (union + eps)).mean()
resnet_path = "/tmp/.cache/torch/checkpoints/"

if(not os.path.exists(resnet_path)):

    os.makedirs(resnet_path)

! cp ../input/resnet18/resnet18.pth /tmp/.cache/torch/checkpoints/resnet18-5c106cde.pth
print(os.listdir("../input/severstal-steel-defect-detection"))

print(os.listdir("../input/masksv1"))
input_folder = Path("../input/severstal-steel-defect-detection")

masks_folder = Path("../input/masksv1/masks")
#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

 

def rle2mask(mask_rle,shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T

def multiplerle2mask(mask_rle_row,shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    if(mask_rle_row['has_rle'] > 0):

        for i in mask_rle_row.index[:-1]:

            class_id = int(i)

            if(not pd.isnull(mask_rle_row[i])):

                s = mask_rle_row[i].split()

                starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

                starts -= 1

                ends = starts + lengths

                for lo, hi in zip(starts, ends):

                    img[lo:hi] = class_id

    return img.reshape(shape).T
def count_masks(x):

#     print(x,x[0])

#     print(pd.isnull(x[0]))

    count = 0

    if not pd.isnull(x[0]) : count = count + 1

    if not pd.isnull(x[1]) : count = count + 1

    if not pd.isnull(x[2]) : count = count + 1

    if not pd.isnull(x[3]) : count = count + 1

    return count
train_df = pd.read_csv(input_folder/"train.csv")
train_df.head()
train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split("_")[1])
train_df['ImageName'] = train_df['ImageId_ClassId'].apply(lambda x: x.split("_")[0])
train_df.head()
train_df_pivot = train_df.pivot(index='ImageName',columns='ClassId',values = 'EncodedPixels')
train_df_pivot['has_rle'] = train_df_pivot.apply(lambda row: count_masks(row), axis = 1)
train_df_pivot
train_df_pivot.has_rle.value_counts()
train_df_pivot[train_df_pivot['has_rle']==1].count()
test_entry = train_df_pivot.iloc[0]
img = open_image(str(input_folder/"train_images"/test_entry.name))
img
def mask_name(name):

    name = Path(name)

    return Path(name.stem+"_mask.png")
masks_folder/mask_name(test_entry.name)
mask = open_mask(masks_folder/mask_name(test_entry.name))
mask
mask.data.unique()
mask2rle(mask.data.numpy()) == test_entry['1']
# mask_paths = Path("./train_masks")
# mask_paths/mask_name(train_df_pivot.iloc[0].name)
# import os

# os.makedirs(mask_paths)
# z = zipfile.ZipFile("masks.zip","w",zipfile.ZIP_DEFLATED)

# for name,row in tqdm(train_df_pivot.iterrows()):

#     temp_mask = multiplerle2mask(row)

#     mask_file_name = mask_name(name)

#     matplotlib.image.imsave(mask_file_name, temp_mask)

#     z.write(mask_file_name)

#     os.remove(mask_file_name)

# z.printdir()

# z.close()

# shutil.make_archive("masks.zip", 'zip', "train_masks")
get_y_fn = lambda x: masks_folder/f'{x.stem}_mask.png'
# # Setting div=True in open_mask

# class SegmentationLabelList(SegmentationLabelList):

#     def open(self, fn): return open_mask(fn, div=True)

    

# class SegmentationItemList(SegmentationItemList):

#     _label_cls = SegmentationLabelList



# # Setting transformations on masks to False on test set

# def transform(self, tfms:Optional[Tuple[TfmList,TfmList]]=(None,None), **kwargs):

#     if not tfms: tfms=(None,None)

#     assert is_listy(tfms) and len(tfms) == 2

#     self.train.transform(tfms[0], **kwargs)

#     self.valid.transform(tfms[1], **kwargs)

#     kwargs['tfm_y'] = False # Test data has no labels

#     if self.test: self.test.transform(tfms[1], **kwargs)

#     return self

# fastai.data_block.ItemLists.transform = transform
data = (SegmentationItemList.from_folder(input_folder/"train_images")

        .split_by_rand_pct()

        .label_from_func(get_y_fn, classes=['0','1','2','3','4'])

        .transform(get_transforms(flip_vert=True), tfm_y=True, size=128)

        .databunch(bs=32, path="/kaggle/working"))

#         .normalize(imagenet_stats))
# data = (SegmentationItemList.from_folder(input_folder/"train_images")

#         #Where to find the data? -> in path_img and its subfolders

#         .split_by_rand_pct()

#         #How to split in train/valid? -> randomly with the default 20% in valid

#         .label_from_func(get_y_fn, classes=['0','1','2','3','4'])

#         #How to label? -> use the label function on the file name of the data

#         .transform(None, tfm_y=True, size=256)

#         # Adding the test folder

# #         .add_test_folder(input_folder/"test_images")

#         #Data augmentation? -> use tfms with a size of 128, also transform the label images

#         .databunch(bs=32))
data
data.show_batch(rows=3, figsize=(7,5))
res_im = data.train_ds[0][0]

mask = data.train_ds[0][1]

act_mask = open_mask(get_y_fn(data.train_ds.items[0]))

# act_mask.data.unique()

# mask.data.unique()
open_image(data.train_ds.items[0])
act_mask
# res_im.show()
# mask.show()
learn = unet_learner(data, models.resnet18,path="/kaggle/working",metrics=[dice])

learn.fit_one_cycle(10,1e-3)

learn.save('mini_train')
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.show_results()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.unfreeze()
learn.fit_one_cycle(10,slice(3e-5))

learn.save('mini_train_unfreeeze')
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.export()
learn.show_results()
# def no_tfms(self, x,**kwargs): return None

# EmptyLabel.apply_tfms = no_tfms
# learn.data.add_test((input_folder/"test_images").ls(),label=None)
# learn.data.test_ds
# Predictions for test set

# preds, _ = learn.get_preds(ds_type=DatasetType.Test)
# preds.shape
# np.unique(preds.numpy(),return_counts=True)
# ys.shape
# np.unique(ys.numpy(),return_counts=True)
# plt.imshow(ys[100][0])
# plt.imshow(preds[120][3])
# open_image(data.valid_ds.items[1])
# open_mask(get_y_fn(data.valid_ds.items[1]))
# test_images = get_image_files(input_folder/"test_images")
# train_images = get_image_files(input_folder/"train_images")