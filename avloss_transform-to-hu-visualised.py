import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import pydicom
dcm_paths = [

    '../input/rsna-str-pulmonary-embolism-detection/train/00c07cd8129d/8877e4d12ce9/08700796d033.dcm',

    '../input/rsna-str-pulmonary-embolism-detection/train/00c73e5a4e16/f41d8a527040/00eb856c55c2.dcm',

    '../input/rsna-str-pulmonary-embolism-detection/train/010f10503133/0fb5dd84a89d/159172ebe2ef.dcm',

    '../input/rsna-str-pulmonary-embolism-detection/train/01b3538d15d6/cd952ef9417e/01bbf9f5eafd.dcm',

    '../input/rsna-str-pulmonary-embolism-detection/train/01d7afb6c23c/9ef17590cf19/1bba75d0969b.dcm',

    '../input/rsna-str-pulmonary-embolism-detection/train/038c6bf912f4/c6d860f22aae/00cbd2da9ab5.dcm',

]
imgs = [pydicom.dcmread(f) for f in dcm_paths]
pa = [img.pixel_array for img in imgs]
row1 = np.concatenate([pa[0],pa[1],pa[2]], axis=1)

row2 = np.concatenate([pa[3],pa[4],pa[5]], axis=1)

rows = np.concatenate([row1, row2], axis=0)
plt.figure(figsize=(20,10))

plt.imshow(rows)
def set_outside_scanner_to_air(raw_pixelarrays):

    # in OSIC we find outside-scanner-regions with raw-values of -2000. 

    # Let's threshold between air (0) and this default (-2000) using -1000

    raw_pixelarrays[raw_pixelarrays <= -1000] = 0

    return raw_pixelarrays
def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)



    images = set_outside_scanner_to_air(images)

    

    # convert to HU

    for n in range(len(slices)):

        

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)

            

        images[n] += np.int16(intercept)

    

    return np.array(images, dtype=np.int16)
trans = [transform_to_hu([img])[0] for img in imgs]
row1 = np.concatenate([trans[0],trans[1],trans[2]], axis=1)

row2 = np.concatenate([trans[3],trans[4],trans[5]], axis=1)

rows = np.concatenate([row1, row2], axis=0)

plt.figure(figsize=(20,10))

plt.imshow(rows)