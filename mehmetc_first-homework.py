# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

data = pd.read_csv('../input/winemag-data_first150k.csv')
data.head()

data = data.drop(['description', 'designation','region_2'], axis=1)

data['price'].mean()
data[data['price'].max() == data['price']]
data[data['price'].min() == data['price']]
data.price.plot(kind = 'line', color = 'red',label = 'price',linewidth=1.5,alpha = 0.8,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Line Plot')           
plt.show()


data.plot(kind='scatter', x='points', y='price',alpha = 0.7,color = 'black')
plt.xlabel('points')              # label = name of label
plt.ylabel('price')
plt.title('Price-Points Scatter Plot') 
plt.show()
data[data['price'] >1000] 
data[data['price'] < 5] 
data[np.logical_and(data['price']>1000, data['points']>99 )]