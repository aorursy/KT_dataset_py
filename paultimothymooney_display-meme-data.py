import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import cv2

import os

from glob import glob

import random
multipleImages = glob('../input/memes/memes/**')

def plot_images(indexStart,indexEnd):

    i_ = 0

    plt.rcParams['figure.figsize'] = (20.0, 20.0)

    plt.subplots_adjust(wspace=0, hspace=0)

    for l in multipleImages[indexStart:indexEnd]:

        im = cv2.imread(l)

        plt.subplot(5, 5, i_+1)

        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')

        i_ += 1
plot_images(0,25)
plot_images(0,25)
plot_images(0,25)
plot_images(0,25)