# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.ndimage.measurements import label

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10,10)

import sys

sys.path.append('../input/siim-acr-pneumothorax-segmentation/')

import mask_functions

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



annotation = pd.read_csv('../input/siimacr-pneumothorax-segmentation-downloaded-file/train-rle.csv')

annotation = annotation.rename(columns = {' EncodedPixels': 'EncodedPixels'})

print('number of strigs in annotaion', len(annotation))

print('number of unique images', len(annotation.drop_duplicates(['ImageId'])))

dublicated = annotation[annotation['ImageId'].duplicated()]

print('number of images with multi blobs ', len(dublicated['ImageId'].unique()))
annotation_blbs = []

cc_blbs = []

for file_name in dublicated['ImageId'].unique():

    annot = dublicated[dublicated['ImageId'] == file_name]['EncodedPixels']

    number_of_annot_blobs = len(annot)

    full_mask = np.zeros(shape=(1024, 1024), dtype=np.bool)

    for rle in annot.tolist():

        full_mask += mask_functions.rle2mask(rle, 1024, 1024).astype(np.bool)

    connected_components_labels, number_of_connected_components = label(full_mask)

    annotation_blbs.append(number_of_annot_blobs)

    cc_blbs.append(number_of_connected_components)

print('number of images with mismatching number cc and annotation', np.sum( [a != c for a, c in zip(annotation_blbs, cc_blbs)]))
count = 0

for file_name in dublicated['ImageId'].unique():

    annot = dublicated[dublicated['ImageId'] == file_name]['EncodedPixels']

    number_of_annot_blobs = len(annot)

    full_mask = np.zeros(shape=(1024, 1024), dtype=np.bool)

    for rle in annot.tolist():

        full_mask += mask_functions.rle2mask(rle, 1024, 1024).astype(np.bool)

    connected_components_labels, number_of_connected_components = label(full_mask)

    if number_of_annot_blobs != number_of_connected_components:

        print('in annotation have ', number_of_annot_blobs, ' blobs')

        print('number of cc ', number_of_connected_components, ' blobs')

        plt.imshow(full_mask)

        plt.show()

        count += 1

        if count >10:

            break