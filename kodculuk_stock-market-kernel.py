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
my_data=pd.read_csv('../input/DJIA_table.csv')
my_data.info()
print(my_data.columns)
my_data.head(10)
my_data.High.plot(kind = 'line', color = 'g',label = 'High',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
my_data.Low.plot(color = 'r',label = 'Low',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Date')              # label = name of label
plt.ylabel('Value')
plt.title('Line Plot')            # title = title of plot
plt.show()
my_data.Volume.plot(kind='hist',color='black',figsize=(10,10),alpha=.7)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Stock Volume According to Date',color='black',size='16')
plt.legend(loc='upper right')
plt.show()
x=my_data['Volume']>539000000
my_data[x]
my_data.plot(kind='scatter',color='purple',alpha=.3,x='Open',y='Close',grid='true',figsize=(15,15))
plt.xlabel('Open',size=25,color='black')
plt.ylabel('Close',size=25,color='black')
plt.title('Open&Close Stock Values')
plt.show()
