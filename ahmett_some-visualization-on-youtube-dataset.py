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
data = pd.read_csv('../input/USvideos.csv')
data.info()
data.head()
data.corr()
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.likes.plot(kind = 'line', color = 'g',label = 'Likes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.comment_count.plot(color = 'r',label = 'Comment Counts',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = likes, y = comment counts
data.plot(kind='scatter', x='likes', y='comment_count',alpha = 0.5,color = 'blue')
plt.xlabel('Likes')              # label = name of label
plt.ylabel('Comment Counts')
plt.title('Like and Comment Counts Scatter Plot')            # title = title of plot
# Scatter Plot 
# x = dislikes, y = comment counts
data.plot(kind='scatter', x='dislikes', y='comment_count',alpha = 0.5,color = 'red')
plt.xlabel('Dislikes')              # label = name of label
plt.ylabel('Comment Counts')
plt.title('Dislike and Comment Counts Scatter Plot')            # title = title of plot
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['likes']>500000) & (data['dislikes']>500000)]
