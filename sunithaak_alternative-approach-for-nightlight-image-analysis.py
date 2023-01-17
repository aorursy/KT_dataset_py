import numpy as np

import matplotlib.pyplot as plt

import cv2
img = cv2.imread(r'../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/China-20200101.tif',0)

def gini(img):

    count = img.size

    coefficient = 2 / count

    indexes = np.arange(1, count + 1)

    weighted_sum = (indexes * img).sum()

    total = img.sum()

    constant = (count + 1) / count

    return coefficient * weighted_sum / total - constant
def lorenz(img):

    # this divides the prefix sum by the total sum

    # this ensures all the values are between 0 and 1.0

    scaled_prefix_sum = img.cumsum() / img.sum()

    # this prepends the 0 value (because 0% of all people have 0% of all wealth)

    return np.insert(scaled_prefix_sum, 0, 0)
import glob

import os

import pandas as pd

rasters = glob.glob(r'../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/*.tif')

Gini_value=[]





for raster in rasters[:10]:

    gini_value= ''

    path, filename = os.path.split(raster)

    img = cv2.imread(raster)

    lorenz_curve = lorenz(img)

    Gini_value.append(gini(lorenz_curve))

    plt.plot(np.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve)

    # plot the straight line perfect equality curve

    plt.plot([0,1], [0,1])

    plt.title(filename)

    plt.show()
print(Gini_value)