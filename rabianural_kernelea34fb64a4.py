# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/cnt_km_year_powerPS_minPrice_maxPrice_avgPrice_sdPrice.csv')

data.info()

data.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

data.columns

# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.km.plot(kind = 'line', color = 'g',label = 'km',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.count.plot(color = 'r',label = 'count',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('km')              # label = name of label

plt.ylabel('count')

plt.title('Line Plot')            # title = title of plot

plt.show()

# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='km', y='count',alpha = 0.5,color = 'red')

plt.xlabel('km')              # label = name of label

plt.ylabel('count')

           # title = title of plot
# Histogram

# bins = number of bar in figure

data.km.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()