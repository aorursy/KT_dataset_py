# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/avocado.csv')
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.3, fmt= '.2f',ax=ax)
plt.show()
data.head(10)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
plt.figure(figsize=(20,10))
data['Total Bags'].plot(kind = 'line', color = 'g',label = 'Total Bags',linewidth=2,alpha = 0.9,grid = True,linestyle = ':')
data['Total Volume'].plot(color = 'r',label = 'Total Volume',linewidth=2, alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Sample Number')              # label = name of label
plt.ylabel('Feature Value of the Sample')
plt.title('Line Plot - Values of Different Features')            # title = title of plot
plt.show()
# Scatter Plot
plt.figure(figsize=(10,10))
plt.scatter(data['Total Volume'], data['Small Bags'], alpha= 0.5, color="r")
plt.xlabel('Total Volume')                                # label = name of label
plt.ylabel('Small Bags')
plt.title('Total Volume Small Bags Scatter Plot')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data['Large Bags'].plot(kind = 'hist',bins = 10,figsize = (12,12))
plt.show()
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))
fig.suptitle('Total Bags and Small Bags Box-and-Whisker Plot', fontsize=16)
# Create an axes instance
# 111 means 1st cell of 1 by 1 grid
ax = fig.add_subplot(111)

## Custom x-axis labels
ax.set_xticklabels(['Total Bags', 'Small Bags'])

# Create the boxplot
# patch_artist= True fills box
bp = ax.boxplot([data['Total Bags'],data['Small Bags']],0, '', patch_artist=True) 

## change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)
    # change fill color
    box.set( facecolor = '#1b9e77' )

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
