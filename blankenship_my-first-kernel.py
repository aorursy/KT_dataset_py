# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/winemag-data-130k-v2.csv')
data.head()
data_filt = data[(data.country== 'US')]

data_filt
data_filt.info()
data_filt.describe()
sns.heatmap(data_filt.corr(),annot = True, linewidths = 5,fmt = '.1f')
data_filt.plot(kind = 'scatter',x = 'points',y = 'price',figsize = (12,12),grid = True,alpha = 0.5)
big_price = data_filt[(data_filt.price > 2000)]

big_price
data_filt.price.plot(kind = 'hist',bins = 50 , figsize = (12,12))

plt.xlabel('price')

plt.title('Price - Frequency')

data_filt.points.plot(kind = 'hist',bins = 20 , figsize = (12,12))

plt.xlabel('points')

plt.title('Points - Frequency')
