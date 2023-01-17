import os

import cv2

import numpy as np

import pandas as pd

import pydicom



from skimage.measure import label,regionprops

from skimage.segmentation import clear_border

import matplotlib.pyplot as plt
# https://www.kaggle.com/currypurin/osic-image-shape-eda-and-preprocess

def crop_image(img: np.ndarray):

    edge_pixel_value = img[0, 0]

    mask = img != edge_pixel_value

    return img[np.ix_(mask.any(1),mask.any(0))]
#d = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00030637202181211009029/100.dcm')

d = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/test/ID00419637202311204720264/21.dcm')
img = crop_image(d.pixel_array)
fig = plt.figure(figsize=(12, 12))



plt.imshow(img)
right_mask = cv2.imread('../input/osic-generalized-lung-mask/mask/right_mask_simetric.jpg', 0)

left_mask = cv2.imread('../input/osic-generalized-lung-mask/mask/left_mask_simetric.jpg', 0)
img = (crop_image(d.pixel_array) + d.RescaleIntercept) / d.RescaleSlope

dim = min(img.shape)

cancer_mask = (img > 10) & (img < 400)

lung_mask = (img > -850) & (img < -600)

lung_mask = cv2.bilateralFilter(lung_mask.astype('float32'),int(dim*0.05),int(dim*0.2),int(dim*0.2)) > 0.1
cancer_mask[~lung_mask] = 0
fig = plt.figure(figsize=(12, 12))



plt.imshow(cancer_mask)

plt.imshow(lung_mask, alpha=0.3)
fig = plt.figure(figsize=(12, 12))



plt.imshow(right_mask, alpha=0.75)

plt.imshow(left_mask, alpha=0.75);
right_mask = cv2.resize(right_mask, lung_mask.shape[::-1]).astype('uint8')

left_mask = cv2.resize(left_mask, lung_mask.shape[::-1]).astype('uint8')
fig = plt.figure(figsize=(12, 12))



plt.imshow(cancer_mask)

plt.imshow(lung_mask, alpha=0.4)

plt.imshow(cv2.resize(right_mask, lung_mask.shape[::-1]), alpha=0.25)

plt.imshow(cv2.resize(left_mask, lung_mask.shape[::-1]), alpha=0.25)
cancer_mask = clear_border(cancer_mask)

lung_mask = clear_border(lung_mask)
fig = plt.figure(figsize=(12, 12))



plt.imshow(cancer_mask)

plt.imshow(lung_mask, alpha=0.4)
from sklearn.metrics import jaccard_score
lung_mask_labeled = label(lung_mask)



fig = plt.figure(figsize=(12, 12))



plt.imshow(lung_mask_labeled)
for i, r in enumerate(regionprops(lung_mask_labeled)):

    _lung_mask = lung_mask.copy()

    m = np.zeros_like(_lung_mask)

    m[r.slice] = 1

    _lung_mask = _lung_mask * m > 0

    riou = jaccard_score(_lung_mask, right_mask > 0, average='micro')

    liou = jaccard_score(_lung_mask, left_mask > 0, average='micro')

    print(f"Region {i}")

    print("\tRight: ", riou, "\n\tLeft: ", liou)

    if liou < 0.1 and riou < 0.1:

        for coordinates in r.coords:                

            lung_mask_labeled[coordinates[0], coordinates[1]] = 0
fig = plt.figure(figsize=(12, 12))



plt.imshow(cancer_mask)

plt.imshow(lung_mask_labeled, alpha=0.5);
fig = plt.figure(figsize=(12, 12))



plt.imshow((crop_image(d.pixel_array) + d.RescaleIntercept) / d.RescaleSlope)

plt.imshow(lung_mask_labeled, alpha=0.5);