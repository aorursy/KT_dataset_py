import os

import numpy as np

import pandas as pd

import pydicom



from skimage.measure import label,regionprops

from skimage.segmentation import clear_border

import matplotlib.pyplot as plt
! pip install git+https://github.com/JoHof/lungmask
import SimpleITK as sitk

from lungmask import mask as lungmask_mask



# credit to: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/183161#1012049

def get_lung_mask(f):

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(f)

    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(f, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()

    series_reader.SetFileNames(sorted_file_names)

    series_reader.MetaDataDictionaryArrayUpdateOn()

    series_reader.LoadPrivateTagsOn()

    image = series_reader.Execute()

    segmentation = (lungmask_mask.apply(image) > 0).astype('uint8')

    return segmentation, image, sorted_file_names





example = '/kaggle/input/rsna-str-pulmonary-embolism-detection/train/b4548bee81e8/ac1aea5d7662/'

out_put = get_lung_mask(example)
out_put[0].shape


import matplotlib.pyplot as plt

plt.imshow(out_put[0][120])

plt.show()