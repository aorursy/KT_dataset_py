import os



import numpy as np

import pandas as pd

import pydicom



from skimage.measure import label,regionprops

from skimage.segmentation import clear_border

import matplotlib.pyplot as plt
d = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/19.dcm')
img = d.pixel_array
fig = plt.figure(figsize=(12, 12))



plt.imshow(img)
img = (img + d.RescaleIntercept) / d.RescaleSlope

img = img < -400
fig = plt.figure(figsize=(12, 12))



plt.imshow(img)
img = clear_border(img)
fig = plt.figure(figsize=(12, 12))



plt.imshow(img)
img = label(img)



fig = plt.figure(figsize=(12, 12))



plt.imshow(img)
areas = [r.area for r in regionprops(img)]

areas.sort()

if len(areas) > 2:

    for region in regionprops(img):

        if region.area < areas[-2]:

            for coordinates in region.coords:                

                img[coordinates[0], coordinates[1]] = 0

img = img > 0
fig = plt.figure(figsize=(12, 12))



plt.imshow(img)