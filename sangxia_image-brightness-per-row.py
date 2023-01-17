# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import cv2

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
from glob import glob
train_names = glob('../input/train_images/*')

test_names = glob('../input/test_images/*')
len(train_names), len(test_names)
plt.figure(figsize=(15,8))



plt.subplot(1,2,1)

arr_row = np.zeros((256,), dtype=float)

for fname in train_names:

    img = cv2.imread(fname)[...,0].astype(float)

    arr_row += np.mean(img, axis=1)

arr_row /= len(train_names)

_ = plt.scatter(np.arange(arr_row.size), arr_row, s=2)

plt.title('Avg brightness per row, train')

plt.subplot(1,2,2)

arr_row = np.zeros((256,), dtype=float)

for fname in test_names:

    img = cv2.imread(fname)[...,0].astype(float)

    arr_row += np.mean(img, axis=1)

arr_row /= len(test_names)

_ = plt.scatter(np.arange(arr_row.size), arr_row, s=2)

plt.title('Avg brightness per row, test')