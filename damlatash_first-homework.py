
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/AppleStore.csv"))

# Any results you write to the current directory are saved as output.
bmyc = pd.read_csv ("../input/AppleStore.csv")
bmyc.info()
bmyc.corr()
import matplotlib.pyplot as plt
import seaborn as sns
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(bmyc.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
bmyc.head(10)
bmyc.columns
bmyc.user_rating.plot(kind = 'line', color = 'g',label = 'user_rating',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
bmyc.price.plot(color = 'r',label = 'price',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
bmyc.plot(kind='scatter', x='user_rating', y='price',alpha = 0.5,color = 'red')
plt.xlabel('user_rating')              # label = name of label
plt.ylabel('price')
plt.title('user_rating price Scatter Plot')
bmyc.price.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
bmyc.price.plot(kind = 'hist',bins = 50)
plt.clf()