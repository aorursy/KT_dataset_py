# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import zipfile

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

import cv2

import torch

from matplotlib import pyplot as plt

from pathlib import Path

import matplotlib

import shutil

from tqdm import tqdm_notebook as tqdm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import PIL
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
train_df = pd.read_csv("/kaggle/input/train.csv")
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
test_entry
image_path = Path("/kaggle/input/train_images")
img = open_image(str(Path("/kaggle/input/train_images")/test_entry.name))
mask_paths = Path("./train_masks")
def mask_name(name):

    name = Path(name)

    return Path(name.stem+"_mask.png")
mask_paths/mask_name(train_df_pivot.iloc[0].name)
import os

os.makedirs(mask_paths)
temp_mask = multiplerle2mask(train_df_pivot.iloc[0])

PIL.Image.fromarray(temp_mask).save('test.png')
mask = open_mask('test.png')
mask2rle(mask.data.numpy()) , train_df_pivot.iloc[0]['1']
z = zipfile.ZipFile("masks.zip","w",zipfile.ZIP_DEFLATED)

for name,row in tqdm(train_df_pivot.iterrows()):

    temp_mask = multiplerle2mask(row)

    mask_file_name = mask_name(name)

    PIL.Image.fromarray(temp_mask).save(mask_file_name)

#     matplotlib.image.imsave(mask_file_name, temp_mask)

    z.write(mask_file_name)

    os.remove(mask_file_name)

z.printdir()

z.close()

# shutil.make_archive("masks.zip", 'zip', "train_masks")
# SegmentationItemList.label_from_func??