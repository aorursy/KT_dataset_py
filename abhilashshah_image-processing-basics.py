# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
image = cv2.imread("../input/test_image_2.png")

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image)
gs = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

plt.imshow(gs,cmap = "Greys_r")
resized = cv2.resize(gs,(256,256))

plt.imshow(resized,cmap = "Greys_r")
print(resized)