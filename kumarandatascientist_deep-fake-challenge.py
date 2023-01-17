# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2 as cv

from matplotlib import pyplot as plt

from tqdm import tqdm

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


print('contents are:', ', '.join(os.listdir('/kaggle/input/deepfake-detection-challenge')))



print(

    'number of  train videos:', len(os.listdir('/kaggle/input/deepfake-detection-challenge/train_sample_videos/')) - 1,

    '\nnumber of test videos: ',  len(os.listdir('/kaggle/input/deepfake-detection-challenge/test_videos/'))

)
train_metadata = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json')



print(train_metadata.head())

print(train_metadata.shape)
train_metadata = train_metadata.T


print(train_metadata.head(10))

print(train_metadata.shape)
train_metadata["label"].value_counts()