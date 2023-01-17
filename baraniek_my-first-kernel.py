# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

data.head()
data.info()
data.corr()
#korelasyon haritası

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.likes.plot(kind = 'line', color = 'g',label = 'likes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.views.plot(color = 'r',label = 'views',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
#scatter



data.plot(kind='scatter',color='blue',x = 'comment_count',y = 'dislikes',alpha = 0.6,grid = True)

plt.xlabel('comment_count')

plt.ylabel('dislikes')

plt.show()
data.views.plot(kind = 'hist',bins = 30,figsize = (12,12))

plt.show()