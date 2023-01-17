%matplotlib inline


data_path = "../input/%s"

train_type_1_path = data_path % "train/Type_1"

train_type_2_path = data_path % "train/Type_2"

train_type_3_path = data_path % "train/Type_3"

test_path = data_path % "test/"
import cv2

import os

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import ndimage

from functools import lru_cache

from itertools import starmap

import numpy as np

import pandas as pd
from functools import partial

from itertools import islice
type_1_img_files = filter(lambda x: x!=".DS_Store", os.listdir(train_type_1_path))

type_2_img_files = filter(lambda x: x!=".DS_Store", os.listdir(train_type_2_path))

type_3_img_files = filter(lambda x: x!=".DS_Store", os.listdir(train_type_3_path))
def get_img_path(name, kind):

    return {

        "type1": "%s/%s" % (train_type_1_path, name),

        "type2": "%s/%s" % (train_type_2_path, name),

        "type3": "%s/%s" % (train_type_3_path, name),

        "test": "%s/%s" % (test_path, name)

    }[kind]
def load_img(name, kind):

    try:

        img = cv2.imread(get_img_path(name, kind))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    except:

        print(name, kind)



def as_gray(img):

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def show_img(imgs, max):

    fig = plt.figure(figsize=(12,8))

    for i, t in zip(range(1, max + 1), imgs):

        ax = fig.add_subplot(int(max / 5), 5, i)

        plt.imshow(t)
def red_filted(img):

    ori = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_red = np.array([100,100,200])

    upper_red = np.array([255,255,255])

    mask = cv2.inRange(ori, lower_red, upper_red)

    return cv2.bitwise_and(img, img, mask=mask)
type_1_imgs = list(islice(map(partial(load_img, kind="type1"), type_1_img_files), 0, 10))

type_2_imgs = list(islice(map(partial(load_img, kind="type2"), type_2_img_files), 0, 10))

type_3_imgs = list(islice(map(partial(load_img, kind="type3"), type_3_img_files), 0, 10))
type_1_gray_imgs = list(map(as_gray, type_1_imgs))

type_2_gray_imgs = list(map(as_gray, type_2_imgs))

type_3_gray_imgs = list(map(as_gray, type_3_imgs))
type_1_mask_imgs = list(map(red_filted, type_1_imgs))

type_2_mask_imgs = list(map(red_filted, type_2_imgs))

type_3_mask_imgs = list(map(red_filted, type_3_imgs))
show_img(type_1_imgs, 5)
show_img(type_1_gray_imgs, 5)
show_img(type_2_imgs, 5)
show_img(type_2_gray_imgs, 5)
show_img(type_2_mask_imgs, 5)
show_img(type_3_imgs, 5)
show_img(type_3_gray_imgs, 5)
show_img(type_3_mask_imgs, 5)
img = type_1_imgs[1]
def xxxx(img):

    ori = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_red = np.array([100,100,100])

    upper_red = np.array([255,255,255])

    mask = cv2.inRange(ori, lower_red, upper_red)

    return cv2.bitwise_and(img, img, mask=mask)







plt.imshow(xxxx(img))