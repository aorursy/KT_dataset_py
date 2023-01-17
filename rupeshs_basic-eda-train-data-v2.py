# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
masks_train = pd.read_csv('../input/train_ship_segmentations_v2.csv')
masks_train.info()
masks_train.head()
#checking train files
TRAIN="../input/train_v2"
file_names=os.listdir(TRAIN)
print("Train files :",len(file_names))
#checking test files
TEST="../input/test_v2"
test_file_names=os.listdir(TEST)
print("Test files :",len(test_file_names))
from PIL import Image
ImageId=file_names[25]
im = Image.open(TRAIN+"/"+ImageId)
im.size
im_test = Image.open(TEST+"/"+test_file_names[5])
im_test.size
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.imshow(im)
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
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
img_masks = masks_train.loc[masks_train['ImageId'] == ImageId, 'EncodedPixels'].tolist()
mask_img = np.zeros((768, 768))
for mask in img_masks:
    mask_img += rle_decode(mask)
plt.imshow(mask_img)
def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax
def get_mask_img(ImageId):
    img_masks = masks_train.loc[masks_train['ImageId'] == ImageId, 'EncodedPixels'].tolist()
    mask_img = np.zeros((768, 768))
    if len(img_masks)==1:
        return mask_img
    for mask in img_masks:
        mask_img += rle_decode(mask)
    return mask_img
fig, axes = plt.subplots(4, 5, figsize=(18, 12))
for i,ax in enumerate(axes.flat):
    imageid=file_names[i+100]
    img=Image.open(TRAIN+"/"+imageid)
    mask=get_mask_img(imageid)
    ax = show_img(img, ax=ax)
    show_img(mask, ax=ax, alpha=0.3)
plt.tight_layout(0.1)
#check for Nan
count=masks_train['EncodedPixels'].isnull().sum()
print("Null mask counts :",count )
print("Train images contatining ships :",len(masks_train)-count )
