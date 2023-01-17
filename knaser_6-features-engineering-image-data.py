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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from skimage.io import imread, imshow



image = imread('../input/water.png', as_gray=True)

imshow(image)
image = imread('../input/water.png', as_gray=True)

image.shape, imshow(image)
features = np.reshape(image, (660*450))



features.shape, features
image = imread('../input/water.png', as_gray=True)

image.shape
image = imread('../input/water.png', as_gray=True)

feature_matrix = np.zeros((660,450)) 

feature_matrix.shape
features = np.reshape(feature_matrix, (660*450)) 

features.shape
#importing the required libraries

import numpy as np

from skimage.io import imread, imshow

from skimage.filters import prewitt_h,prewitt_v

import matplotlib.pyplot as plt

%matplotlib inline



#reading the image 

image = imread('../input/water.png', as_gray=True)



#calculating horizontal edges using prewitt kernel

edges_prewitt_horizontal = prewitt_h(image)

#calculating vertical edges using prewitt kernel

edges_prewitt_vertical = prewitt_v(image)



imshow(edges_prewitt_vertical, cmap='gray')