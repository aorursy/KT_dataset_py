# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_video = pd.read_csv("../input/FRvideos.csv")
data_video.info()
data_video.columns
data_video.corr()
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(data_video.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data_video.head(10)
data_video.tail(10)
data_video.likes.plot(kind='line', color='g', label='Likes',grid=True, linewidth=1, alpha=0.5, linestyle=':')
data_video.dislikes.plot(color='r', label='Dislikes',grid=True, linewidth=1, alpha=0.5, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
data_video.plot(kind='scatter', color='red', x='dislikes',y='views', alpha=0.5)
plt.xlabel('dislikes')
plt.ylabel('views')

data_video.likes.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
