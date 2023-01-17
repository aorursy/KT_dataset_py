import numpy as np

import pandas as pd

import pydicom

import os

import matplotlib.pyplot as plt
ds = pydicom.dcmread('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_000039fa0.dcm')

ds
print('pixel_array:', ds.pixel_array)

print('center:',ds.pixel_array[206:306,206:306])

print('dimensions:', ds.pixel_array.shape)

plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
rescale_intercept = ds[('0028','1052')].value

rescale_slope = ds[('0028','1053')].value



rescaled_hu = rescale_slope * ds.pixel_array + rescale_intercept
print('Transition zone:')

print(ds.pixel_array[75:85,65:75])
print('Air-to-cranium transition zone:')

print(rescaled_hu[105:115,160:170])
plt.imshow(rescaled_hu, cmap=plt.cm.bone)
y_min = 0

y_max = 255

window_center = ds[('0028','1050')].value

window_width = ds[('0028','1051')].value



windowed_hu = rescaled_hu.copy()

min_val = window_center - window_width / 2

max_val = window_center + window_width / 2



# we want pixels with min_val to have score zero

windowed_hu = windowed_hu - min_val;



# we want pixels with the max value to have a score of 255

windowed_hu = windowed_hu * y_max / max_val



# we want to contrain all other values

windowed_hu = np.clip(windowed_hu, y_min, y_max)



## have a look

plt.imshow(windowed_hu, cmap=plt.cm.bone)