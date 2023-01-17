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
import numpy as np



from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline



from skimage.io import imread

from skimage.io import imshow



from skimage.io import imread



import pandas as pd # data processing
img = imread('/kaggle/input/painting.png', as_gray=True)



print(img)
imshow(img)





painting.shape, imshow(painting)





print(painting.shape)

painting.shape, imshow(img)

img.shape, imshow(img)

print(greyscale.shape)
features = np.reshape(img, (80*80))
features.shape, features
features.shape, features

feature_matrix = np.ones([80,80])

feature_matrix.shape

for i in range(img.shape[0]):

    for j in range(0,greyscale.shape[1]):

        feature_matrix[i][j] = ((float(greyscale[i,j,0]) +  float(greyscale[i,j,1]) + float(greyscale[i,j,2]))/3)

features = np.reshape(feature_matrix, (25600,))

features.shape

f= np.array(img)

l= f.T

print(l)

a = np.average(img)

df = np.array(a)