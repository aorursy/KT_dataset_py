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
import os

import cv2

import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/pku-autonomous-driving/train.csv')
train_df.info()
train_expanded = pd.concat([train_df, train_df['PredictionString'].str.split(' ', expand=True).astype(float)], axis=1)
train_expanded = train_expanded.drop(['PredictionString'], axis=1)
columns = []

for col in train_expanded:

    if isinstance(col, int):

        if col % 7 == 0:

            columns.append(f'modeltype_{col//7:d}')

        elif col % 7 == 1:

            columns.append(f'yaw_{col//7:d}')

        elif col % 7 == 2:

            columns.append(f'pitch_{col//7:d}')

        elif col % 7 == 3:

            columns.append(f'roll_{col//7:d}')

        elif col % 7 == 4:

            columns.append(f'x_{col//7:d}')

        elif col % 7 == 5:

            columns.append(f'y_{col//7:d}')

        elif col % 7 == 6:

            columns.append(f'z_{col//7:d}')            

    else:

        columns.append(col)

train_expanded.columns = columns
train_expanded['CarAmount'] = (train_df['PredictionString'].str.split(' ').apply(len) // 7).astype(int)
col = train_expanded.columns.tolist()

col = col[0:1] + col[-1:] + col[1:-1]

train_expanded = train_expanded[col]
train_expanded = train_expanded.sort_values(by=['ImageId'])
train_expanded['CarAmount'].plot(kind='hist', figsize=(15, 3), bins=100, title='Distribution of cars in each image')
train_image_path = '../input/pku-autonomous-driving/train_images/'

train_mask_path = '../input/pku-autonomous-driving/train_masks/'
train_images = next(os.walk(train_image_path))[2]

train_images.sort()
train_masks = next(os.walk(train_mask_path))[2]

train_masks.sort()
sample_image = cv2.cvtColor(cv2.imread(train_image_path + train_images[0]), cv2.COLOR_BGR2RGB)

print(sample_image.shape)
sample_mask = cv2.cvtColor(cv2.imread(train_mask_path + train_masks[0]), cv2.COLOR_BGR2RGB)

print(sample_mask.shape)
plt.figure(figsize=(15,15))

plt.imshow(sample_image)

plt.imshow(sample_mask, alpha=0.65)