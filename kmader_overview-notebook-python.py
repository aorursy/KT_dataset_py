import os

from skimage.io import imread

from glob import glob

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
with np.load('../input/flo_image_1.npz') as im_data:

    image_dict = dict(zip(im_data['image_ids'], im_data['image_stack']))

print('Loaded',len(image_dict), 'images')
# show a few test images

fig, m_axs = plt.subplots(2,3)

for (ax1, ax2), (img_name, img_data) in  zip(m_axs.T, image_dict.items()):

    ax1.imshow(img_data)

    ax1.set_title(img_name)

    ax2.hist(img_data.ravel())
time_df = pd.read_csv('../input/data141110.csv')

time_df['path'] = time_df['Image.No.'].map(lambda x: "141110A3.%04d" % (x))

time_df['loaded'] = time_df['path'].map(lambda x: x in image_dict)

valid_time_df = time_df.query('loaded')

valid_time_df.sample(3)
valid_time_df['mean_intensity'] = valid_time_df['path'].map(lambda x: np.mean(image_dict[x]))

valid_time_df['std_intensity'] = valid_time_df['path'].map(lambda x: np.std(image_dict[x]))
fig, ax1 = plt.subplots(1,1)

ax1.plot(valid_time_df['Time.hrs.'], valid_time_df['mean_intensity'])

ax1.plot(valid_time_df['Time.hrs.'], valid_time_df['std_intensity'])

ax1.set_xlabel('Time (hours)')

ax1.set_ylabel('Intensity (mean)')
# if we show the data mod 24 do we see stronger trends 

fig, ax1 = plt.subplots(1,1)

ax1.plot(np.mod(valid_time_df['Time.hrs.'], 24), valid_time_df['mean_intensity'], 'b.')

ax1.plot(np.mod(valid_time_df['Time.hrs.'], 24), valid_time_df['std_intensity'], 'g.')

ax1.set_xlabel('Time (hours)')

ax1.set_ylabel('Intensity (mean)')