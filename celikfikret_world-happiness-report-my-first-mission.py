# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
plt.show()

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2015.csv')
data.info()
data.corr()

#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

data.head(10)
# tail shows last 5 rows
data.tail()
data.columns

data.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split())>1) else each for each in data.columns]
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Generosity.plot(kind = 'line', color = 'g',label = 'Generosity',linewidth=1,alpha = 0.7,grid = True,linestyle = ':')
data.Freedom.plot(color = 'r',label = 'Freedom',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Health_(Life', y='Economy_(GDP',alpha = 0.7,color = 'red')
plt.xlabel('Health_(Life')              # label = name of label
plt.ylabel('Freedom')
plt.title('Health_Life Economy_GDP Scatter Plot')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.Freedom.plot(kind = 'hist',bins = 50, figsize = (10,5))
plt.show()
# For example max HP is 255 or min defense is 5
data.describe() #ignore null entries
data.boxplot(column='Happiness_Score',by = 'Happiness_Rank')
plt.show()
data1 = data.loc[:,["Trust_(Government","Freedom","Health_(Life"]]
data1.plot(subplots = True)
plt.show()
data1.plot(figsize = (10,5))
plt.show()
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
m.bluemarble(scale=0.5);

