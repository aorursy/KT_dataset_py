# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import colors

import tifffile



colormap = np.zeros([3,3])

colormap[1] = colors.to_rgba_array('0.1')[:, :-1]

mask = np.zeros([2, 2, 3])



ary = tifffile.imread('../input/sixteen_band/6100_2_2_M.tif')



np.percentile(ary, [2, 98], axis=(1, 2))