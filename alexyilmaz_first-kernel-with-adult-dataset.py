# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as p # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math as mt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
p.__version__ #Which Version Pandas we use

np.__version__

sns.__version__
df=p.read_csv("../input/adult.csv")#Import Adult Data
df.info()       # memory footprint and datatypes
df.all()
df.head()       # first five rows
df.tail()       # last five rows
df.describe()   # calculates measures of central tendency
df.sample(5)    # random sample of rows
df.shape        # number of rows/columns in a tuple
Male=(df[df.sex==" Male"]) # Filter By sex columns for Male
Male.sex.unique()
Female=(df[df.sex==" Female"]) # Filter By sex columns for Female
Female.sex.unique()
df.sex.unique()
df.columns
df.groupby('race').count() # group by race all data 
fig = plt.figure(figsize=(20,15))
cols = 5
rows = mt.ceil(float(df.shape[1]) / cols)
for i, column in enumerate(df.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if df.dtypes[column] == np.object:
        df[column].value_counts().plot(kind="bar", axes=ax)
    else:
        df[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
df.capital_gain.plot(kind = 'line', color = 'g',label = 'capital_gain',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.capital_loss.plot(color = 'r',label = 'hours_per_week',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()

# Scatter Plot 
# x = capital_gain, y = hours_per_week
df.plot(kind='scatter', x='capital_gain', y='hours_per_week',alpha = 0.5,color = 'red')
plt.xlabel('capital_gain')              # label = name of label
plt.ylabel('hours_per_week')
plt.title('capital_gain-hours_per_week Scatter Plot')            # title = title of plot
plt.show()

#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(Female.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Histogram
# bins = number of bar in figure
df.age.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
