# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import scipy.ndimage

from scipy.misc.pilutil import Image

import skimage

import pylab



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

os.listdir("../input/train/ALB")



# Any results you write to the current directory are saved as output.

fish_alb = Image.open('../input/train/ALB/img_07075.jpg') 

pylab.imshow(fish_alb)
