# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels as sm
import math as mt
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.tree as tree
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/bitcoin.csv")
df.info
df.corr()
df.shape
df["GAP"]=df.Open-df.Close
df2=df[df.GAP>0]
df2.shape
df3=(df2[df2.Timestamp>=1514764800])
df3.shape
fig = plt.figure(figsize=(20,15))
cols = 5
rows = mt.ceil(float(df3.shape[1]) / cols)
for i, column in enumerate(df3.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if df3.dtypes[column] == np.object:
        df3[column].value_counts().plot(kind="bar", axes=ax)
    else:
        df3[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
df3.Open.plot(kind = 'line', color = 'R',label = 'Open',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df3.Close.plot(color = 'B',label = 'Close',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('After_201801')            # title = title of plot
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
df.Open.plot(kind = 'line', color = 'g',label = 'Open',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
df.Close.plot(color = 'r',label = 'Close',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('All Data')            # title = title of plot
plt.show()


# Scatter Plot 
# x = attack, y = defense
df3.plot(kind='scatter', x='GAP', y='Open',alpha = 0.5,color = 'red')
plt.xlabel('GAP')              # label = name of label
plt.ylabel('Open')
plt.title('Open-GAP')            # title = title of plot
plt.show()


# Scatter Plot 
# x = attack, y = defense
df3.plot(kind='scatter', x='GAP', y='Close',alpha = 0.5,color = 'B')
plt.xlabel('GAP')              # label = name of label
plt.ylabel('Close')
plt.title('Close-GAP')            # title = title of plot
plt.show()

# Scatter Plot 
# x = attack, y = defense
df3.plot(kind='scatter', x='Timestamp', y='High',alpha = 0.5,color = 'Blue')
plt.xlabel('High')              # label = name of label
plt.ylabel('Timestamp')
plt.title('High-Timestamp')            # title = title of plot
plt.show()


# Scatter Plot 
# x = attack, y = defense
df3.plot(kind='scatter', x='Volume_(Currency)', y='Close',alpha = 0.5,color = 'red')
plt.xlabel('Volume_(Currency)')              # label = name of label
plt.ylabel('Close')
plt.title('Close-Volume_(Currency)')            # title = title of plot
plt.show()



#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df3.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


df3.Close.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()

