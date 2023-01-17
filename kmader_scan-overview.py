import numpy as np # for manipulating 3d images

import pandas as pd # for reading and writing tables

import h5py # for reading the image files

import skimage # for image processing and visualizations

import sklearn # for machine learning and statistical models

import os # help us load files and deal with paths
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
train_df = pd.read_csv('../input/train.csv')

train_df.head(5) # show the first 5 lines
train_df['age_years'].hist(bins=10) # make a histogram of the ages of patients
train_df['age_group'] = train_df['age_years'].map(lambda age: 'old' if age>60 else 'young') 

train_df['age_group'].value_counts() # show how many of each we have
sample_scan = train_df.iloc[0] # just take the first row

print(sample_scan)

# turn the h5_path into the full path

full_scan_path = os.path.join('..', 'input','train', sample_scan['h5_path'])

# load the image using hdf5

with h5py.File(full_scan_path, 'r') as h:

    image_data = h['image'][:][:, :, :, 0] # we read the data from the file

print(image_data.shape, 'loaded')
# show the middle slice

plt.imshow(image_data[image_data.shape[0]//2, :, :])
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(image_data[image_data.shape[0]//2, :, :], cmap='gray')

ax2.imshow(image_data[:, image_data.shape[1]//2, :], cmap='gray')

ax3.imshow(image_data[:, :, image_data.shape[2]//2], cmap='gray')
brain_montage = skimage.util.montage(image_data)

plt.imshow(brain_montage)
fig, m_axs = plt.subplots(2, 3, figsize=(15, 10))

for (group_name, c_rows), (ax1, ax2, ax3) in zip(train_df.groupby('age_group'), m_axs):

    full_scan_path = os.path.join('..', 'input','train', c_rows['h5_path'].iloc[0])

    with h5py.File(full_scan_path, 'r') as h:

        cur_image_data = h['image'][:][:, :, :, 0] # we read the data from the file

    ax1.imshow(cur_image_data[cur_image_data.shape[0]//2, :, :], cmap='gray')

    ax1.set_title('{} scan'.format(group_name))

    ax2.imshow(cur_image_data[:, cur_image_data.shape[1]//2, :], cmap='gray')

    ax3.imshow(cur_image_data[:, :, cur_image_data.shape[2]//2], cmap='gray')