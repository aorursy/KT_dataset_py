import os

import numpy as np

import pandas as pd

import pydicom



from skimage.measure import label,regionprops

from skimage.segmentation import clear_border

import matplotlib.pyplot as plt
! pip install git+https://github.com/SoufianeDataFan/lungmask
d = pydicom.dcmread('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/19.dcm')

img = d.pixel_array

fig = plt.figure(figsize=(12, 12))

plt.imshow(img)
from lungmask import mask

import SimpleITK as sitk



def get_mask(filename, plot_mask=False, return_val=False): 

    # Let's an example of a CT scan

    input_image = sitk.ReadImage(filename)

    mask_out = mask.apply(input_image)[0]  #default model is U-net(R231)

    if plot_mask: 

        fig = plt.figure(figsize=(12, 12))

        plt.imshow(mask_out)

    if return_val:

        return mask_out
img_mask = get_mask('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/19.dcm',

                    plot_mask=True,

                    return_val=True)
