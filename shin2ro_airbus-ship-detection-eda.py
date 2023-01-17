import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from skimage.io import imread

import os
print(os.listdir('../input'))
segmentations = pd.read_csv('../input/train_ship_segmentations.csv')
segmentations.head()
print('Number of images')
print(f'- containing ships     : {len(segmentations[segmentations.EncodedPixels.isna() == False].ImageId.unique())}')
print(f'- not containing ships : {len(segmentations[segmentations.EncodedPixels.isna()].ImageId.unique())}')
image_ids = segmentations[segmentations.EncodedPixels.isna() == False].sample(6, random_state=123).ImageId
image_ids
image_id = image_ids.iat[0]
img = imread(f'../input/train/{image_id}')
print(f'Image id   : {image_id}')
print(f'Image shape: {img.shape}')

plt.axis('off')
plt.imshow(img)
plt.show()
def rle_decode(encoded_pixels, shape):
    s = encoded_pixels.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(shape).T
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, image_id in enumerate(image_ids):
    img = imread(f'../input/train/{image_id}')
    mask_shape = img.shape[:-1]
    mask = np.zeros(mask_shape)
    
    encoded_pixels_list = segmentations[segmentations.ImageId == image_id].EncodedPixels.tolist()
    for encoded_pixels in encoded_pixels_list:
        mask += rle_decode(encoded_pixels, mask_shape)
        
    row = i // 2
    col = i * 2 % 4
    axes[row][col].axis('off')
    axes[row][col+1].axis('off')
    axes[row][col].imshow(img)
    axes[row][col+1].imshow(mask)
    
plt.tight_layout(h_pad=0, w_pad=0)
plt.show()
image_ids = segmentations[segmentations.EncodedPixels.isna()].sample(8, random_state=123).ImageId
image_ids
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, image_id in enumerate(image_ids):
    img = imread(f'../input/train/{image_id}')
    
    row = i // 4
    col = i % 4
        
    axes[row][col].axis('off')
    axes[row][col].imshow(img)
    
plt.tight_layout(h_pad=0, w_pad=0)
plt.show()
