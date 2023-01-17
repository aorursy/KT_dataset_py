# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
HEIGHT = 137

WIDTH = 236

df = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
train_labels = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

train_labels.head()
# most represented in train data

train_labels.groupby('grapheme_root').agg('count').reset_index()[['grapheme_root','grapheme']].sort_values('grapheme', ascending=False).head()
# least represented in train data

train_labels.groupby('grapheme_root').agg('count').reset_index()[['grapheme_root','grapheme']].sort_values('grapheme').head()
## We can now filter a small df with the graphene we want to further explore (lets try graphene = 72)

filtered_df = train_labels[train_labels['grapheme_root'] == 72]

img_ids =  filtered_df['image_id'].values

print("There are {} images in train0 of grapehene root = 72".format(len(filtered_df)))
## filter the images to the ones of interest:

filtered_imgs = df[df['image_id'].isin(img_ids)]

#check how many we have in our subset of images (train0)

print("Number of images in train0 with graphene root = 72: {}".format(len(filtered_imgs)))
images_72 = filtered_imgs.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
f, ax = plt.subplots(4, 4, figsize=(16, 8))

ax = ax.flatten()

for i in range(16):

    ax[i].imshow(images_72[i], cmap='Greys')
## We can now filter a small df with the graphene we want to further explore (lets try graphene = 73)

filtered_df = train_labels[train_labels['grapheme_root'] == 73]

img_ids =  filtered_df['image_id'].values

print("There are {} images in train0 of grapehene root = 73".format(len(filtered_df)))

######

## filter the images to the ones of interest:

filtered_imgs = df[df['image_id'].isin(img_ids)]

#check how many we have in our subset of images (train0)

print("Number of images in train0 with graphene root = 73: {}".format(len(filtered_imgs)))

images_73 = filtered_imgs.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
f, ax = plt.subplots(4, 4, figsize=(16, 8))

ax = ax.flatten()

for i in range(16):

    ax[i].imshow(images_73[i], cmap='Greys')