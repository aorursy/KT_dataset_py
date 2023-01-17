# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import misc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
i = misc.ascent()

plt.gray()

plt.imshow(i)
image = np.copy(i)

x = image.shape[0]

y = image.shape[1]
#filters = [[-1,-2,-1],[0,0,0],[1,2,1]]

filters = [[-1,-2,-1],[-2,0,2],[1,2,1]]

#filters = [[-1,-1,-1],[0,0,0],[1,1,1]]



weight =0.5

for j in range(1,x-1):

    for k in range(1,y-1):

        convolution = 0.0

        convolution = convolution + (i[j-1 , k-1] * filters[0][0])

        convolution = convolution + (i[j , k-1] * filters[0][1])

        convolution = convolution + (i[j+1 , k-1] * filters[0][2])

        convolution = convolution + (i[j-1 , k] * filters[1][0])

        convolution = convolution + (i[j , k] * filters[1][1])

        convolution = convolution + (i[j+1 , k] * filters[1][2])

        convolution = convolution + (i[j-1 , k+1] * filters[2][0])

        convolution = convolution + (i[j , k+1] * filters[2][1])

        convolution = convolution + (i[j+1 , k+1] * filters[2][2])

        convolution = convolution *weight

        if convolution <0:

            convolution=0

        if convolution >255:

            convolution =255

        image[j,k] = convolution    
plt.gray()

plt.imshow(image)
new_x = int(x/2)

new_y = int(y/2)



newImage = np.zeros((new_x, new_y))



for j in range(0,x,2):

    for k in range(0,y,2):

        pixel=[]

        pixel.append(image[j,k])

        pixel.append(image[j+1,k])

        pixel.append(image[j,k+1])

        pixel.append(image[j+1,k+1])

        newImage[int(j/2) , int(k/2)] = max(pixel)
plt.gray()

plt.imshow(newImage)