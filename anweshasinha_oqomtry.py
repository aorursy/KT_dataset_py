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
import random

import os

from matplotlib.pyplot import imshow

import numpy as np

from PIL import Image

from os import listdir



files = listdir("/kaggle/input/Warwick QU Dataset (Released 2016_07_08)")



# our folder path containing some images

folder_path = '/kaggle/input/Warwick QU Dataset (Released 2016_07_08)/'

# the number of file to generate

num_files_desired = 1000



# loop on all files of the folder and build a list of files paths

images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

#print(images)

import pandas as pd

import matplotlib.pyplot as plt

train_path1 = "/kaggle/input/Warwick QU Dataset (Released 2016_07_08)/Grade.csv"

train = pd.read_csv(train_path1)

train.head()   

#img = Image.open('/kaggle/input/Warwick QU Dataset (Released 2016_07_08)/train_32.bmp')

for i in range(train.shape[0]):

     if files[i]!='Grade.csv':

        img = Image.open('/kaggle/input/Warwick QU Dataset (Released 2016_07_08)/'+files[i])

        img = np.array(img)

        plt.imshow(img)

        plt.show()

    

#img = Image.open('/kaggle/input/Warwick QU Dataset (Released 2016_07_08)/train_32.bmp')

#Flipping The Image

for i in range(train.shape[0]):

    if files[i]!='Grade.csv':

        img = Image.open('/kaggle/input/Warwick QU Dataset (Released 2016_07_08)/'+files[i])

        flipped_img = np.fliplr(img)

        plt.imshow(flipped_img)

        plt.show()

# Shifting Left

HEIGHT = 5

WIDTH = 5

for i in range(HEIGHT, 1, -1):

  for j in range(WIDTH):

     if (i < HEIGHT-20):

       img[j][i] = img[j][i-20]

     elif (i < HEIGHT-1):

       img[j][i] = 0

plt.imshow(img)

plt.show()