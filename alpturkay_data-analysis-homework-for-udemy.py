# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # for visualization 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv') # import data
data.info() # info about data
data.columns
data.head(10)
data.describe() # Statistical values for dataset
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Scatter Plot 
# x = compactness mean, y = smoothness mean
data.plot(kind='scatter', x='compactness_mean', y='smoothness_mean',alpha = 0.5,color = 'b')
plt.xlabel('Compactness mean')             
plt.ylabel('Smoothness mean')
plt.title('compactness-mean smoothness-mean Scatter Plot')            
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.fractal_dimension_mean.plot(kind = 'line', color = 'y',label = 'fractal_dimention_mean',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.smoothness_mean.plot(color = 'r',label = 'smoothness_mean',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Fractal Dimention Mean vs Smoothness Mean')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.radius_mean.plot(kind = 'hist',y = "Radius Mean",bins = 50,figsize = (10,10))
plt.show()