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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='latin-1',low_memory=False)
#data.info()
#data.corr()
"""f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()"""
#data.head(15)
#data.columns
"""data.iyear.plot(kind = 'line', color = 'g',label = 'iyear',linewidth=5,alpha = 0.5,grid = True,linestyle = ':')
data.imonth.plot(kind = 'line', color = 'r',label = 'imonth',linewidth=3,alpha = 0.7,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Line Plot')            
plt.show()"""

data.plot(kind='scatter', x='country', y='region',alpha = 0.5,color = 'red')
plt.xlabel('Country')              # label = name of label
plt.ylabel('Region')
plt.title('Country-Region Scatter Plot')

data.iday.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()

series = data['country']        # data['Defense'] = series
print(type(series))
data_frame = data[['country']]  # data[['Defense']] = data frame
print(type(data_frame))

data[np.logical_and(data['iyear']>1970, data['region']>2 )]