import cv2

import numpy as np

import matplotlib.pyplot as plt

import glob

%matplotlib inline



sunflowers_path = glob.glob("../input/flowers/flowers/sunflower/*.jpg")

sunflower_bgr = cv2.imread(sunflowers_path[1])

plt.figure(figsize=(10,10))

plt.imshow(sunflower_bgr);
sunflower_rgb = cv2.cvtColor(sunflower_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))

plt.imshow(sunflower_rgb);
def plot_channels(img):

    '''plot each image channel'''

    f, axes = plt.subplots(1,3, figsize = (15,15))

    i = 0

    for ax in axes:

        ax.imshow(img[:,:,i], cmap = "gray")

        i+=1



plot_channels(sunflower_rgb)
sunflower_hsv = cv2.cvtColor(sunflower_rgb, cv2.COLOR_RGB2HSV)

plot_channels(sunflower_hsv)