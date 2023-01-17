# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

train_data = os.listdir("../input/train/")

test_data = os.listdir("../input/test/")



import keras

import cv2



from keras.layers import Conv2D

from keras.preprocessing.image import img_to_array, array_to_img
x = []

for image in train_data:

    x.append(cv2.imread(os.path.join("../input/train", image)))
x.shape
