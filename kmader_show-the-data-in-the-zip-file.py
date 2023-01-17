import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread

import matplotlib.pyplot as plt

from glob import glob

import os
tif_file_df = pd.DataFrame([{'path': filepath} for filepath in glob('../input/tiff_images/*.tif')])

tif_file_df['file'] = tif_file_df['path'].map(os.path.basename)

tif_file_df['Contrast'] = tif_file_df['file'].map(lambda x: bool(x.split('_')[-2]))

tif_file_df['Age'] = tif_file_df['file'].map(lambda x: int(x.split('_')[3]))

tif_file_df.sample(4)
tif_file_df['Age'].hist()
test_row = list(tif_file_df.sample(1).T.to_dict().values())[0] # grab a random row

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 4))

test_row_img = imread(test_row['path'])

ax1.matshow(test_row_img)

ax1.set_title('Slice\nAge: {Age}, Contrast: {Contrast}'.format(**test_row))

ax1.axis('off')

ax2.hist(test_row_img.ravel())

ax2.set_title('Histogram\nAge: {Age}, Contrast: {Contrast}'.format(**test_row))