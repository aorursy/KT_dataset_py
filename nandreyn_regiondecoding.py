import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import cv2

import skimage as ski

import os

TRAIN_FOLD, TEST_FOLD = '../input/understanding_cloud_organization/train_images', '../input/understanding_cloud_organization/test_images'

train = pd.read_csv('../input/understanding_cloud_organization/train.csv')
train.head()
image_name = os.listdir(TRAIN_FOLD)[0]

test_image = ski.io.imread(os.path.join(TRAIN_FOLD, image_name))

print(f'Image : {image_name}')

plt.imshow(test_image)
train.columns
test_image_regions = train[train['Image_Label'].str.contains(image_name)]

test_image_regions.head()
def mask_to_image_decoding(image_shape, mask_string):

    if str(mask_string) == 'nan':

        return np.zeros(image_shape).astype(np.uint8)

    if not ((isinstance(image_shape, np.ndarray) and image_name.ndim != 2) or len(image_name) != 2):

        raise ValueError('Expected 2D image size')



    pairs = list(map(int, mask_string.split(' ')))

    mask_pairs = [

        (x, y) for x, y in zip(pairs[::2], pairs[1::2]) 

    ]

    mask = np.zeros(image_shape)

    

    for start, length in mask_pairs:

        mask[np.unravel_index(list(range(start, start + length)), image_shape, order='F')] = 255

        

    return mask.astype(np.uint8)

   
def plot_image_and_mask(image, string_mask):

    shp = list(image.shape)[:-1]

    mask = mask_to_image_decoding(shp, string_mask)

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    cv2.drawContours(img_gray, contours[0], -1, (255), 8)

    to_display = np.hstack([img_gray, mask])

    plt.figure(figsize=(12, 6))

    plt.imshow(to_display, cmap='gray')
plot_image_and_mask(test_image, test_image_regions['EncodedPixels'].iloc[0])
plot_image_and_mask(test_image, test_image_regions['EncodedPixels'].iloc[1])
plot_image_and_mask(test_image, test_image_regions['EncodedPixels'].iloc[2])
plot_image_and_mask(test_image, test_image_regions['EncodedPixels'].iloc[3])