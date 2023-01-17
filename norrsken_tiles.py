import numpy as np

import os

import pandas as pd

import sys



import cv2

import matplotlib.pyplot as plt

import openslide

from PIL import Image

import skimage.io

from tqdm.notebook import tqdm
TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'

sz = 64 # Dimension of each tile.

N = 49 # Number of tiles.
train_csv = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

images = train_csv['image_id']
def tile(img):

    result = []

    shape = img.shape

    

    '''Calculating the padding required in Horizontal and Vertical direction respectively'''

    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

    

    '''np.pad is used to pad the image. the second positional argument represents number of rows/columns to be padded, ie.

        [[top_pad, bottom_pad], [left_pad, right_pad],[front_pad, back_pad]], with value 255'''

    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                constant_values=255)

    

    '''Reshaping the image into blocks of 128* 128

    So, the img becomes 5 dimensional, and dimensions representthe following:

    (num_vertical_blocks, height_each_block, num_horizontal_blocks, width_each_block, dimension)'''

    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

    

    '''Reshaping them into (num_horizontal_blocks x num_vertical_blocks) blocks of 128x128 in 3 dimensions'''

    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

    

    '''If the number of blocks is less than the 16 or N, then pad remaining blocks with value 255 (White pixels).'''

    if len(img) < N:

        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

        

    '''Now get top N blocks from the image which have the lowest sum (which means they have fewer white pixels and hence more information).'''

    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

    

    '''Filter out the N blocks and then stack them into a single image of dimensions (512x512x3) with N tiles concatenated together.'''

    img = img[idxs]

    img = np.vstack((np.hstack((img[:7])),np.hstack((img[7:14])),np.hstack((img[14:21])),np.hstack((img[21:28])),np.hstack((img[28:35])),np.hstack((img[35:42])),np.hstack((img[42:49]))))

    return img
'''Apply the tile() to all the images and convert them into Concatenated tile images. We'll use then use these images to train our models.'''

for x in tqdm(images):

    '''Get the image of lowest resolution from the MultiImage.'''

    img = skimage.io.MultiImage(os.path.join(TRAIN,f'{x}.tiff'))[-1]

    img = tile(img)



    img = Image.fromarray(img, 'RGB')

    img.save(f'{x}.png')

img = skimage.io.MultiImage(os.path.join(TRAIN,f'003d4dd6bd61221ebc0bfb9350db333f.tiff'))[-1]

print(img.shape)

fig = plt.figure(figsize = (10, 10))

plt.imshow(img)

plt.show()
img = tile(img)

print(img.shape)

fig = plt.figure(figsize = (10, 10))

plt.imshow(img)

plt.show()