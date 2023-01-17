# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from imageio import imread
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
path_train = '../input/train/'
path_test = '../input/test/'

train_segmentation = pd.read_csv('../input/train_ship_segmentations.csv')
test_segmentation = pd.read_csv('../input/test_ship_segmentations.csv')
# now lets check the format of our sample submission file
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
# referecens for two kernel
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# ref: https://www.kaggle.com/inversion/run-length-decoding-quick-start

from skimage.morphology import label
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

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

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ref https://www.kaggle.com/kmader/baseline-u-net-model-part-1
def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768,768))
    for mask in img_mask:
        all_masks += rle_decode(mask)
    return all_masks
# now lets do some EDA on our training data
train_segmentation.head()
ImageId = '00021ddc3.jpg'
img = imread('../input/train/'+ImageId)
img_mask = train_segmentation.loc[train_segmentation['ImageId'] == ImageId, 'EncodedPixels']
print(img_mask)
print('Number os images with the same given ImageId:', len(img_mask))
img_mask = img_mask.tolist()
# Looking at this ImageId we can see that is not a rule that in the train data the id are unique,
all_masks_img0 = masks_as_image(img_mask)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 16))

# 'turn off' the axis for all subplots
for i in range(3):
    axes[i].axis('off')

# plot the original image    
axes[0].imshow(img)
# plot the masks of the image
axes[1].imshow(all_masks_img0)
# plot the image and all masks with an alpha equals 0.4
axes[2].imshow(img)
axes[2].imshow(all_masks_img0, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
fig_e, axes_e = plt.subplots(1, 2, figsize=(12,12))
axes_e[0].axis('off')
axes_e[1].axis('off')
axes_e[0].set_title('Image$_0$')
axes_e[0].imshow(all_masks_img0)
# encode the image loaded earlier
rle_img1 = multi_rle_encode(img)
# read the img from the rle_img1
img1 = rle_decode(rle_img1)
axes_e[1].set_title('Image$_1$')
axes_e[1].imshow(img1)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
