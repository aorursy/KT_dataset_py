#libraries

import os

import sys

import gc



import numpy as np

import pandas as pd

import seaborn as sns



from pathlib import Path, PosixPath

from PIL import Image, ImageChops

from matplotlib import pyplot as plt



%matplotlib inline
#paths

ROOT = Path(".").resolve().parents[0]

INPUT_ROOT = ROOT/"input"

RAW_DATA = INPUT_ROOT/"alaska2-image-steganalysis"

TRAIN_COVER = RAW_DATA/"Cover"

TRAIN_JMIPOD = RAW_DATA/"JMiPOD"

TRAIN_JUNIWARD = RAW_DATA/"JUNIWARD"

TRAIN_UERD =  RAW_DATA/"UERD"

TEST = RAW_DATA/"Test"
def read_image(image_id: str, image_dir: PosixPath):

    with open(image_dir / image_id, "rb") as fr:

        img = Image.open(fr)

        img.load()

    return img
def compare_image(image_id: str):

    cover = read_image(image_id, TRAIN_COVER)

    steganography = [

        ["JMiPOD", read_image(image_id, TRAIN_JMIPOD)],

        ["JUNIWARD", read_image(image_id, TRAIN_JUNIWARD)],

        ["UERD", read_image(image_id, TRAIN_UERD)],

    ]

    

    fig = plt.figure(figsize=(20, 20))

    

    sz_x = sz_y = 4

    

    ax_cov = fig.add_subplot(sz_x, sz_y, 4 * 0 + 1)

    ax_cov.set_title("Cover", fontsize=22)

    ax_cov.imshow(cover)

    

    ax_sub = fig.add_subplot(sz_x, sz_y, 4 * 0 + 2)

    ax_sub.set_title("{}".format(steganography[0][0]), fontsize=22)  

    sub_arr = np.asarray(cover) - np.asarray(steganography[0][1])

    ax_sub.imshow(Image.fromarray(sub_arr.astype("uint8")))

    

    ax_sub = fig.add_subplot(sz_x, sz_y, 4 * 0 + 3)

    ax_sub.set_title("{}".format(steganography[1][0]), fontsize=22)  

    sub_arr = np.asarray(cover) - np.asarray(steganography[1][1])

    ax_sub.imshow(Image.fromarray(sub_arr.astype("uint8")))

    

    ax_sub = fig.add_subplot(sz_x, sz_y, 4 * 0 + 4)

    ax_sub.set_title("{}".format(steganography[2][0]), fontsize=22)  

    sub_arr = np.asarray(cover) - np.asarray(steganography[2][1])

    ax_sub.imshow(Image.fromarray(sub_arr.astype("uint8")))

def compare_crop_image(image_id: str, crop_area):

    cover = read_image(image_id, TRAIN_COVER)

    cover = cover.crop(crop_area)

    steganography = [

        ["JMiPOD", read_image(image_id, TRAIN_JMIPOD)],

        ["JUNIWARD", read_image(image_id, TRAIN_JUNIWARD)],

        ["UERD", read_image(image_id, TRAIN_UERD)],

    ]

    fig = plt.figure(figsize=(20, 20))

    

    sz_x = sz_y = 4

    

    ax_cov = fig.add_subplot(sz_x, sz_y, 4 * 0 + 1)

    ax_cov.set_title("Cover", fontsize=22)

    ax_cov.imshow(cover)

    

    ax_sub = fig.add_subplot(sz_x, sz_y, 4 * 0 + 2)

    ax_sub.set_title("{}".format(steganography[0][0]), fontsize=22)  

    sub_arr = np.asarray(cover) - np.asarray(steganography[0][1].crop(crop_area))

    ax_sub.imshow(Image.fromarray(sub_arr.astype("uint8")))

    

    ax_sub = fig.add_subplot(sz_x, sz_y, 4 * 0 + 3)

    ax_sub.set_title("{}".format(steganography[1][0]), fontsize=22)  

    sub_arr = np.asarray(cover) - np.asarray(steganography[1][1].crop(crop_area))

    ax_sub.imshow(Image.fromarray(sub_arr.astype("uint8")))

    

    ax_sub = fig.add_subplot(sz_x, sz_y, 4 * 0 + 4)

    ax_sub.set_title("{}".format(steganography[2][0]), fontsize=22)  

    sub_arr = np.asarray(cover) - np.asarray(steganography[2][1].crop(crop_area))

    ax_sub.imshow(Image.fromarray(sub_arr.astype("uint8")))

    

#sorted images in the dir as they are unsorted

train_image_ids = sorted(os.listdir(TRAIN_COVER))
for i in range(0,10):

    compare_image(train_image_ids[i])
for i in range(0,10):

    compare_crop_image(train_image_ids[i], (0, 0, 40, 40))