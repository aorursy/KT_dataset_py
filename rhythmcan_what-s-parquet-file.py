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

        fullpath = os.path.join(dirname, filename)

        print('{}:{} MB'.format(fullpath, round(os.path.getsize(fullpath) / (1024.0 ** 2), 1)))



# Any results you write to the current directory are saved as output.
train_image_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')
train_image_0.shape
train_image_0.head()
train_image_0.index
train_image_data = train_image_0.drop('image_id', axis=1)
train_image_data.head()
IMAGE_ROW = 137 

IMAGE_COLUMN = 236



img0 = train_image_data[0:1].values[0].reshape([IMAGE_ROW, IMAGE_COLUMN])

img0
from matplotlib import pylab as plt



plt.imshow(img0)