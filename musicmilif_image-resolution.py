import os
import numpy as np
import pandas as pd
import cv2

import matplotlib.pylab as plt
%matplotlib inline
def resolution_detect(df, data_path, train=True):
    """
    Get resolution for each seismic images. Threshold may set at 0.06

    Args:
        df: Training or testing dataframe with `id` column
        data_path: The folder where you stored seismic images
        train: Boolean, True if training dataframe, False if testing dataframe

    Returns:
        Pandas DataFrame with new column named 'resolution'
    """
    folder = 'train' if train else 'test'
    data_path = os.path.join(data_path, folder)

    resolution = []
    for id_ in df['id'].values:
        image = cv2.imread(os.path.join(data_path, 'images/{}.png'.format(id_)), 
                           cv2.IMREAD_GRAYSCALE)
        image = np.array(image / 255, np.float32)
        resolution.append(cv2.Laplacian(image, cv2.CV_32F).std())

    df['resolution'] = resolution

    return df
DATA_DIR = '../input/'

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), 
                       usecols={'id': str})
train_df = resolution_detect(train_df, DATA_DIR)
train_df.head()
# Show images with high resolution
high_resolution_ids = train_df['id'].loc[train_df['resolution']>0.1].sample(n=3)
high_resolution_imgs = []
for id_ in high_resolution_ids:
    img = cv2.imread(os.path.join(DATA_DIR, 'train/images/{}.png'.format(id_)), 
                     cv2.IMREAD_GRAYSCALE)
    
    high_resolution_imgs.append(img)

plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(high_resolution_imgs[0], cmap='gray')
plt.subplot(132)
plt.imshow(high_resolution_imgs[1], cmap='gray')
plt.subplot(133)
plt.imshow(high_resolution_imgs[2], cmap='gray')
plt.show()
# Show images with low resolution
low_resolution_ids = train_df['id'].loc[train_df['resolution']<0.05].sample(n=3)
low_resolution_imgs = []
for id_ in low_resolution_ids:
    img = cv2.imread(os.path.join(DATA_DIR, 'train/images/{}.png'.format(id_)), 
                     cv2.IMREAD_GRAYSCALE)
    
    low_resolution_imgs.append(img)

plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(low_resolution_imgs[0], cmap='gray')
plt.subplot(132)
plt.imshow(low_resolution_imgs[1], cmap='gray')
plt.subplot(133)
plt.imshow(low_resolution_imgs[2], cmap='gray')
plt.show()

