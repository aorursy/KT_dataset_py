# imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2 

import skimage.io
# dataset

PATH = '../input/prostate-cancer-grade-assessment/'

IMG_PATH = PATH + 'train_images/'

data = pd.read_csv(PATH + 'train.csv')

data.head()
# helper function

def read_image(image_path, level=1):

    image = skimage.io.MultiImage(image_path)

    image = image[level]

    return image





# credits to Rohit Singh

pen_marked_images = [

    'fd6fe1a3985b17d067f2cb4d5bc1e6e1',

    'ebb6a080d72e09f6481721ef9f88c472',

    'ebb6d5ca45942536f78beb451ee43cc4',

    'ea9d52d65500acc9b9d89eb6b82cdcdf',

    'e726a8eac36c3d91c3c4f9edba8ba713',

    'e90abe191f61b6fed6d6781c8305fe4b',

    'fd0bb45eba479a7f7d953f41d574bf9f',

    'ff10f937c3d52eff6ad4dd733f2bc3ac',

    'feee2e895355a921f2b75b54debad328',

    'feac91652a1c5accff08217d19116f1c',

    'fb01a0a69517bb47d7f4699b6217f69d',

    'f00ec753b5618cfb30519db0947fe724',

    'e9a4f528b33479412ee019e155e1a197',

    'f062f6c1128e0e9d51a76747d9018849',

    'f39bf22d9a2f313425ee201932bac91a',

]

def remove_pen_marks(img):

    

    # Define elliptic kernel

    kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    

    # Convert image to gray scale and mask out background

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_mask = np.where(img_gray < 210, 1, 0).astype(np.uint8)

    

    # Reshape red channel into 1-d array, aims to mask most of the pen marks

    img_r = np.reshape(img[:, :, 0], (-1,))

    img_r = img_r[np.where(img_r < 255)[0]]

    img_r_mask = (img[:, :, 0] < np.median(img_r)-50).astype(np.uint8)



    # When computing the pen mark mask, some tissue gets masked as well,

    # thus needing to erode the mask to get rid of it. Then some dilatation is 

    # applied to capture the "edges" of the "gradient-like"/non-uniform pen marks

    img_r_mask = cv2.erode(img_r_mask, kernel5x5, iterations=3)

    img_r_mask = cv2.dilate(img_r_mask, kernel5x5, iterations=5)

    

    # Combine the two masks

    img_r_mask = 1 - img_r_mask

    img_mask = img_mask * img_r_mask

    

    # There might still be some gaps/holes in the tissue, here's an attempt to 

    # fill those gaps/holes

    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel5x5, iterations=1)

    img_mask = cv2.dilate(img_mask, kernel5x5, iterations=1)

    contours, _ = cv2.findContours(img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        cv2.drawContours(img_mask, [contour], 0, 1, -1)

    

    # Some final touch

    img_mask = cv2.erode(img_mask, kernel5x5, iterations=3)

    img_mask = cv2.dilate(img_mask, kernel5x5, iterations=1)

    img_mask = cv2.erode(img_mask, kernel5x5, iterations=2)

    

    # Mask out pen marks from original image

    img = img * img_mask[:, :, np.newaxis]

    

    return img, img_mask





fig, axes = plt.subplots(

    len(pen_marked_images), 3, figsize=(10, 10*len(pen_marked_images))

)



for i, ID in enumerate(pen_marked_images):

    

    img_path = PATH + 'train_images/' + ID + '.tiff'

    img = read_image(img_path, level=1)

    img2, mask = remove_pen_marks(img)

    

    axes[i, 0].imshow(img)

    axes[i, 0].set_title('Original Image', fontsize=16)

    axes[i, 1].imshow(mask)

    axes[i, 1].set_title('No-pen Mask', fontsize=16)

    axes[i, 2].imshow(img2)

    axes[i, 2].set_title('No-pen Image', fontsize=16)

    for ax in axes[i]:

        ax.axis('off')

    
def remove_pen_marks(img):

    

    # Define elliptic kernel

    kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    

    # use cv2.inRange to mask pen marks (hardcoded for now)

    lower = np.array([0, 0, 0])

    upper = np.array([200, 255, 255])

    img_mask1 = cv2.inRange(img, lower, upper)



    # Use erosion and findContours to remove masked tissue (side effect of above)

    img_mask1 = cv2.erode(img_mask1, kernel5x5, iterations=4)

    img_mask2 = np.zeros(img_mask1.shape, dtype=np.uint8)

    contours, _ = cv2.findContours(img_mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        x, y = contour[:, 0, 0], contour[:, 0, 1]

        w, h = x.max()-x.min(), y.max()-y.min()

        if w > 100 and h > 100:

            cv2.drawContours(img_mask2, [contour], 0, 1, -1)

    # expand the area of the pen marks

    img_mask2 = cv2.dilate(img_mask2, kernel5x5, iterations=3)

    img_mask2 = (1 - img_mask2)

    

    # Mask out pen marks from original image

    img = cv2.bitwise_and(img, img, mask=img_mask2)

    

    img[img == 0] = 255

    

    return img, img_mask1, img_mask2





fig, axes = plt.subplots(

    len(pen_marked_images), 4, figsize=(12, 10*len(pen_marked_images))

)



for i, ID in enumerate(pen_marked_images):

    

    img_path = PATH + 'train_images/' + ID + '.tiff'

    img = read_image(img_path, level=1)

    img2, mask1, mask2 = remove_pen_marks(img)

    

    axes[i, 0].imshow(img)

    axes[i, 0].set_title('Original Image', fontsize=16)

    axes[i, 1].imshow(mask1)

    axes[i, 1].set_title('Mask 1', fontsize=16)

    axes[i, 2].imshow(mask2)

    axes[i, 2].set_title('Mask 2', fontsize=16)

    axes[i, 3].imshow(img2)

    axes[i, 3].set_title('New Image', fontsize=16)

    for ax in axes[i]:

        ax.axis('off')

    