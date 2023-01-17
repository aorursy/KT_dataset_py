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
#print(os.listdir("../input/flowers/flowers/sunflower"))
from keras.preprocessing.image import load_img
a= np.array(load_img("../input//flowers/flowers/sunflower/8014734302_65c6e83bb4_m.jpg"))
print(a.shape)
import matplotlib.pyplot as plt
plt.imshow(a)
b=a[:,:,0]
c=a[:,:,1]*256
d=a[:,:,2]*(256*256)
e=b+c+d
print(e)
print(e.shape)
print(e.min(),e.max())