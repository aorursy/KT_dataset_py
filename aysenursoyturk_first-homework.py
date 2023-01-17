# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
data1 = pd.read_csv('../input/breastCancer.csv')
data1.info()
data1.corr()
data1.head(10)
data1.tail(10)
data1.columns
data1.clump_thickness.plot(kind = 'line', color = 'g',label = 'clumpthickness',linewidth=1,alpha = 0.75,grid = True,linestyle = ':')
data1.epithelial_size.plot(kind = 'line', color = 'r',label = 'epithelialsize',linewidth=1,alpha = 0.75,grid = True,linestyle = '-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot')
plt.show()
data1.plot(kind='scatter',x='bland_chromatin',y='mitoses',alpha=0.75,color='red')
plt.show()

data1.marginal_adhesion.plot(kind = 'hist',bins = 50,figsize = (10,10))
plt.show()