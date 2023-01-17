# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.data import imread
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Dataset
import os
print(os.listdir('../input'))
print(os.listdir('../input/test_v2')[:5])
print(os.listdir('../input/train_v2')[:5])
# Peak at the training data, EncodedPixels format
train = pd.read_csv('../input/train_ship_segmentations_v2.csv')
train.head()
def subplot_of_images(samples, n_cols=5, image_size_inches=20):
    '''Quickly plot images to matplotlib subplots'''
    n_rows = int(math.ceil(len(samples) / float(n_cols)))

    # Create matplotlib subplots
    fig, ax = plt.subplots(n_rows, n_cols, sharex='col', sharey='row')
    fig.set_size_inches(image_size_inches, image_size_inches)

    # Set the images to subplots
    for i, imgid in enumerate(samples.ImageId):
        col = i % n_cols
        row = i // n_cols

        path = Path('../input/train_v2') / '{}'.format(imgid)
        img = imread(path)

        ax[row, col].imshow(img)
# Plot the images with ships
n_sample = 16
sample = train[~train.EncodedPixels.isna()].sample(n_sample)
subplot_of_images(sample, n_cols=4, image_size_inches=10)
# Plot the images without ships
n_sample = 16
sample = train[train.EncodedPixels.isna()].sample(n_sample)
subplot_of_images(sample, n_cols=4, image_size_inches=10)
# Histogram of training data with/without ships
ships = train[~train.EncodedPixels.isna()].ImageId.unique()
noships = train[train.EncodedPixels.isna()].ImageId.unique()

plt.bar(['Ships', 'No Ships'], [len(ships), len(noships)])
plt.ylabel('Number of Images');
# Decode run-length encoding to rectangular B&W image (mask)
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
# View a few of the masks from the training data, which should highlight tanker outlines
masks = pd.read_csv('../input/train_ship_segmentations_v2.csv')
masks.head(30)
def plot_image_and_rle_mask(img, img_masks):
    '''Quickly plot image along side the run-length encoding mask(s)'''

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768))
    for mask in img_masks:
        all_masks += rle_decode(mask)

    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks, cmap='gray')
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()
# Compare a few images with tanker ships, the mask, and both overlaid
img_ids = []
img_ids.append('00113a75c.jpg')
img_ids.append('00b0fa633.jpg')

for img_id in img_ids:
    img = imread('../input/train_v2/' + img_id)
    img_masks = masks.loc[masks['ImageId'] == img_id, 'EncodedPixels'].tolist()
    plot_image_and_rle_mask(img, img_masks)