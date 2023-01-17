# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input"))

views = pd.read_csv("../input/youtube-new/USvideos.csv")

# views.corr()



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(views.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()



views.likes.plot(kind = 'line', color = 'g',label = 'Likes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

views.comment_count.plot(color = 'r',label = 'Comment Counts',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()



views.plot(kind='scatter', x='likes', y='comment_count',alpha = 0.5,color = 'blue')

plt.xlabel('Likes')              # label = name of label

plt.ylabel('Comment Counts')

plt.title('Like and Comment Counts Scatter Plot')    





# Any results you write to the current directory are saved as output.