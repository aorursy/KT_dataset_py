import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

tqdm.pandas()

from pathlib import Path

import PIL
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
train_df.head()
del train_df['bbox_x1']

del train_df['bbox_x2']

del train_df['bbox_y1']

del train_df['bbox_y2']
print('total class : {}'.format(train_df['class'].nunique()))
sns.countplot(train_df['class'])
train_df['class'].value_counts().head()
train_df['class'].value_counts().tail()
def get_img_size(img_id, path):

            

    img = PIL.Image.open(f'{path}/{img_id}')

    

    return img.size
img_dir_path = Path('../input/train/')
train_df['shape'] = train_df['img_file'].progress_apply(lambda x : get_img_size(x, img_dir_path))
width_list = [x[0] for x in train_df['shape'].values]

height_list = [x[1] for x in train_df['shape'].values]
train_df['width'] = width_list

train_df['height'] = height_list

del (width_list, height_list)

del train_df['shape']
train_df[train_df['width'] == train_df['width'].max()]
train_df[train_df['width'] == train_df['width'].min()]
train_df[train_df['height'] == train_df['height'].max()]
train_df[train_df['height'] == train_df['height'].min()]
plt.figure(figsize=(14,6))

plt.subplot(121)

sns.distplot(train_df['width'], kde=False, label='width')

plt.legend()

plt.title('Image width', fontsize=15)



plt.subplot(122)

sns.kdeplot(train_df['width'], label='width')

plt.legend()

plt.title('Image width KDE Plot', fontsize=15)
plt.figure(figsize=(14,6))

plt.subplot(121)

sns.distplot(train_df['height'], kde=False, label='height')

plt.legend()

plt.title('Image width', fontsize=15)



plt.subplot(122)

sns.kdeplot(train_df['height'], label='height')

plt.legend()

plt.title('Image height KDE Plot', fontsize=15)
train_df['aspect_ratio'] = train_df['height'] / train_df['width']
plt.figure(figsize=(14,6))

plt.subplot(121)

sns.distplot(train_df['aspect_ratio'], kde=False, label='aspect_ratio')

plt.legend()

plt.title('Image aspect_ratio', fontsize=15)



plt.subplot(122)

sns.kdeplot(train_df['aspect_ratio'], label='aspect_ratio')

plt.legend()

plt.title('Image aspect_ratio KDE Plot', fontsize=15)