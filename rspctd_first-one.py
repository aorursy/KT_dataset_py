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
data1=pd.read_csv('../input/winemag-data_first150k.csv')
data1.info()
data1.corr()
data1.head(12)
data1.tail(12)
data1.columns
#Line Plot
data1.points.plot(kind='line',color='blue',label='points',linewidth=1,alpha=100,grid=True,linestyle=':')
data1.price.plot(kind='line',color='red',label='price',linewidth=1,alpha=200,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()

#Scatter Plot
data1.plot(kind='scatter',x='points',y='price',alpha=0.5,grid=True,label='points and price')
plt.legend(loc='upper center')
plt.xlabel('points')
plt.ylabel('price')
plt.title('Points and Price')
plt.show()


data1.columns
data1.points.plot(kind='hist',bins=45,figsize=(12,12))
plt.show()
x=(data1['price']<100)&(data1['points']>90)&(data1['country']=="Turkey")
data1[x]

data1[x].plot(kind='scatter',x='points',y='price',color='y',grid=True)
plt.show()
#ı have completed this homework but I am pretty sure that there are couple of things that I need to do on it. I will be appreciated, if you can give me some advise.Thanks again
