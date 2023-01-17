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
import numpy as np

import pandas as pd

import random

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import os
N_SAMPLE = 10

IMAGES_DIR = "../input/malaria/malaria/images/"
# load a few images to view

imagePaths = os.listdir(IMAGES_DIR)
# show images



fig = plt.figure(figsize= (20, 20))

cols = 1

rows = 10



for i in range(N_SAMPLE):

    filePath = os.path.join(IMAGES_DIR, random.choice(imagePaths))

    ax1 = fig.add_subplot(rows, cols, i + 1)

    img = mpimg.imread(filePath)

    plt.imshow(img)

    

plt.show()