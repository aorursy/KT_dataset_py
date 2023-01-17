

import os

import sys

import random

import warnings



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from tqdm import tqdm_notebook, tnrange

from skimage.io import imread, imshow, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import array_to_img, img_to_array, load_img

from skimage.feature import canny

from skimage.filters import sobel,threshold_otsu, threshold_niblack,threshold_sauvola

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

from skimage.segmentation import mark_boundaries

from scipy import signal

from pathlib import Path





import cv2

from PIL import Image

import pdb

from tqdm import tqdm

import seaborn as sns

import os 

from glob import glob





import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



data=pd.read_csv('../input/train_ship_segmentations_v2.csv')
data.head()
PATH='../input/'
train_imgs=os.listdir(PATH+'train_v2')

test_imgs=os.listdir(PATH+'test_v2')
masks = pd.read_csv(os.path.join('../input/',

                                 'train_ship_segmentations_v2.csv'))

print(masks.shape[0], 'masks found')

print(masks['ImageId'].value_counts().shape[0])

masks.head()
from sklearn.model_selection import train_test_split

unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')

train_ids, valid_ids = train_test_split(unique_img_ids, 

                 test_size = 0.3, 

                 stratify = unique_img_ids['counts'])

train_df = pd.merge(masks, train_ids)

valid_df = pd.merge(masks, valid_ids)

print(train_df.shape[0], 'training masks')

print(valid_df.shape[0], 'validation masks')
train_imgs[:5]
test_imgs[:5]
data = data.reset_index()

data['ship_count'] = data.groupby('ImageId')['ImageId'].transform('count')

print(data['ship_count'].describe())
df = df.ImageId()
def get_filename(image_id, image_type):

    check_dir = False

    if "Train" == image_type:

        data_path = train_imgs

    elif "mask" in image_type:

        data_path = masks

    elif "Test" in image_type:

        data_path = test_imgs

    else:

        raise Exception("Image type '%s' is not recognized" % image_type)



    if check_dir and not os.path.exists(data_path):

        os.makedirs(data_path)



    return os.path.join(data_path, "{}".format(image_id))



def get_image_data(image_id, image_type, **kwargs):

    img = _get_image_data_opencv(image_id, image_type, **kwargs)

    img = img.astype('uint8')

    return img



def _get_image_data_opencv(image_id, image_type, **kwargs):

    fname = get_filename(image_id, image_type)

    img = cv2.imread(fname)

    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img





def rle_decode(mask_rle, shape=(768, 768)):

   

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T 
sample = masks[~masks.EncodedPixels.isna()].sample(9)



fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')

fig.set_size_inches(20, 20)



for i, imgid in enumerate(sample.ImageId):

    col = i % 3

    row = i // 3

    

    path = Path('../input/train_v2') / '{}'.format(imgid)

    img = imread(path)

    

    ax[row, col].imshow(img)
from skimage.filters import gaussian,laplace
ImageId = '0005d01c8.jpg'



img = imread('../input/train_v2/' + ImageId)

img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()



# Take the individual ship masks and create a single mask array for all ships

all_masks = np.zeros((768, 768))

for mask in img_masks:

    all_masks += rle_decode(mask)



fig, axarr = plt.subplots(1, 3, figsize=(15, 40))

axarr[0].axis('off')

axarr[1].axis('off')

axarr[2].axis('off')

axarr[0].imshow(img)

axarr[1].imshow(all_masks)

axarr[2].imshow(img)

axarr[2].imshow(all_masks, alpha=0.4)

plt.tight_layout(h_pad=0.1, w_pad=0.1)

plt.show()