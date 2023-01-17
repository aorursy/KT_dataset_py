# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/tmdb_5000_movies.csv") #firstly reading variable
data.info() # data information
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)
plt.show()
data.columns # columns name
data.head(5)# top 5 list
data.budget.plot(kind = 'line', color = 'y',label = 'budget',linewidth=2,alpha = 0.7,grid = True,linestyle = '--')
data.vote_average.plot(kind = "line", color = 'r',label = 'vote_average',linewidth=4, alpha = 0.9,grid = True,linestyle = ':')

plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
data.runtime.plot(kind = 'hist',bins = 15,figsize = (8,8))
plt.show()
plt.clf()
x = data['budget']>300000000
y = data ["budget"]>200000000
data[x],data[y]
data[np.logical_and(data['runtime']>150, data['runtime']<180 )]