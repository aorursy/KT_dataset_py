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

print(check_output(["ls", "../input"]).decode("utf8"))

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/fake.csv")
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.likes.plot(kind = 'line', color = 'g',label = 'likes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.shares.plot(color = 'r',label = 'shares',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='participants_count', y='replies_count',alpha = 0.5,color = 'red')

plt.xlabel('Participants')              # label = name of label

plt.ylabel('Replies')

plt.title('Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.spam_score.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.title("Spam Score")

plt.show()
data = pd.read_csv('../input/fake.csv')

series = data['spam_score']        # data['spam_score'] = series

print(type(series))

data_frame = data[['spam_score']]  # data[['spam_score']] = data frame

print(type(data_frame))

# 1 - Filtering Pandas data frame

x = data['spam_score']==0     # There are 12999 rows that have 0 value for spam_score

data[x]
# 2 - Filtering pandas with logical_and

# There are 9619 rows that have 0 value for spam_score and 1 value for participants_count

data[np.logical_and(data['spam_score']==0, data['participants_count']==1 )]