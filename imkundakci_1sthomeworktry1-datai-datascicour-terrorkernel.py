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
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',engine='python') 
# write "engine = 'python'" to prevent the occurence of encoding error
data.info()
a=0
for i in data.columns:
    print (i)
    a=a+1                    # Of course there are 135 entries that can be seen from data.info()
    
print(a)
data.tail(10)

data.head(10)
data.corr()
# correlation map
f, ax = plt.subplots(figsize=(50,50))
sns.heatmap(data.corr(),annot=True, linewidths =.5, fmt='.1f',ax = ax)
plt.show()
data.filter(items=['iyear','targtype1','targsubtype1_txt','country_txt','country'])
# Line Plot
data.targtype1.plot(kind = 'line', color = 'g',label = 'target type', linewidth=1, alpha = 0.5, grid = True, linestyle = ':')
data.latitude.plot(color = 'r', label = 'latitude', linewidth=1, alpha = 0.5, grid = True, linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
#data.iyear.plot(kind = 'hist',bins = 50,figsize = (12,12))
#plt.show()
data.targtype1.plot(kind = 'hist',figsize = (12,12))
plt.show()