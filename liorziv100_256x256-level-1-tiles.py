import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import os

import cv2

import random

import skimage.io

from tqdm import tqdm
BASE_PATH = "/kaggle/input/prostate-cancer-grade-assessment"

TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train_images")

MASKS_PATH = os.path.join(BASE_PATH, "train_label_masks")



train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))



# 0 is highest quality, 1 is x4 smaller, 2 is x16 smaller

IMAGES_LEVEL = 1

TILE_SIZE = 256

TILE_NUM = 32
"""

Taken from https://www.kaggle.com/iafoss/panda-16x128x128-tiles

"""

def tile(img, mask):

    sz = TILE_SIZE

    N = TILE_NUM

    

    result = []

    shape = img.shape

    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                constant_values=255)

    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                constant_values=0)

    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)

    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

    if len(img) < N:

        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)

        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

    img = img[idxs]

    mask = mask[idxs]

    for i in range(len(img)):

        result.append({'img':img[i], 'mask':mask[i], 'idx':i})

    return result
def read_img(image_id):

    image = skimage.io.MultiImage(f"{TRAIN_IMG_PATH}/{image_id}.tiff")

    return image[IMAGES_LEVEL]



def read_mask(image_id):

    image = skimage.io.MultiImage(f"{MASKS_PATH}/{image_id}_mask.tiff")

    if len(image) <= 1:

        shape = read_img(image_id).shape

        return np.zeros(shape=shape, dtype=np.uint8)

    return image[IMAGES_LEVEL]
# chosen_im_i = random.randint(0, len(train)-1)

chosen_im_i = 100

chosen_img = read_img(train.at[chosen_im_i, "image_id"])

chosen_img_mask = read_mask(train.at[chosen_im_i, "image_id"])

tiles_example = tile(chosen_img, chosen_img_mask)



fig, axis = plt.subplots(8,8, figsize=(25,20))

for i in range(4):

    for j in range(8):

        t = tiles_example[j + 8*i]

        axis[2*i, j].imshow(t['img'])

        axis[2*i+1, j].imshow(t['mask']*100)

plt.show()
fig, axis = plt.subplots(1,2, figsize=(25,10))

axis[0].imshow(chosen_img)

axis[1].imshow(chosen_img_mask * 100)

plt.show()
%%time



os.makedirs("images", exist_ok=True)

os.makedirs("masks", exist_ok=True)



LIMIT = 3

for image_id in train.image_id[:LIMIT]:

    img = read_img(image_id)

    mask = read_mask(image_id)

    tiles = tile(img, mask)



    for i in range(TILE_NUM):

        plt.imsave(fname=f"images/{image_id}_{i}.png", arr=tiles[i]['img'], format="png")

        plt.imsave(fname=f"masks/{image_id}_{i}.png", arr=tiles[i]['mask'], format="png")