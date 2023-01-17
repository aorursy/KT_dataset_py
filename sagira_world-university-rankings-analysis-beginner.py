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
data = pd.read_csv('../input/cwurData.csv')  #reading the data
data.info()
data.head(5)  #first 5
data.tail(5) #last 5
data.columns
data.corr()
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.2f', ax=ax)
plt.show()
dF2015 = data[data['year'] == 2015]

# Line plotting
dF2015.broad_impact.plot(kind = 'line', color = 'g', label = 'broad_impact', linewidth = 1, alpha = 0.6, grid = True, linestyle = ':')
dF2015.world_rank.plot(kind = 'line', color = 'blue', label = 'world_rank', linewidth = 1, alpha = 0.6, grid = True, linestyle = '-.')
plt.legend()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()

# Scatter Plot
dF2015.plot(kind = 'scatter', x = 'broad_impact', y = 'world_rank',color = 'b', alpha = 0.5)
plt.xlabel('broad_impact')
plt.ylabel('world_rank')
plt.title('broad_impact  world_rank scatter plot ')
plt.show()
#Histogram    To see the frequency of one column's variables.
data.publications.plot(kind = 'hist', bins = 100, figsize = (12,12),color = 'b',alpha = 0.8)
plt.show()
dFUSA = dF2015[dF2015.country == 'USA']
dFUK = dF2015[dF2015.country == 'United Kingdom']
dFJAPAN = dF2015[dF2015.country == 'Japan']
dFSWITZERLAND = dF2015[dF2015.country == 'Switzerland']
dFISRAEL = dF2015[dF2015.country == 'Israel']
score_means =[dFUSA[0:5].score.mean(),dFUK[0:5].score.mean(),dFJAPAN[0:5].score.mean(),dFSWITZERLAND[0:5].score.mean(),dFISRAEL[0:5].score.mean()]
plt.bar(dF2015.country.unique()[0:5] , score_means[0:5], width=0.5, color='r',alpha = 0.7)
plt.show()