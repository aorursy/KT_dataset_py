# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Import opencv and matplot libraries

import cv2

import matplotlib.pyplot as plt
# Check if we can find the data

train_dir = "../input/300_train/300_train/"

f = os.listdir(train_dir)

print(f[:10])
# Read the labels

data_file = train_dir + "newTrainLabels.csv"

df = pd.read_csv(data_file)

print(df.head())

print(df.tail())
df['image'][12345]
# Read image #12345 and show it

a = cv2.imread(train_dir + df['image'][12345] + '.jpeg')

print('Pixel (min, max):', np.min(a), np.max(a))

plt.imshow(a)

plt.title(df['image'][12345] + '.jpeg')

plt.show()