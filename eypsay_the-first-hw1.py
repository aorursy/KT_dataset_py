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
#by eypsay

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")
#data.info()
data.corr()
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax)
plt.show()
data.head(15)
data.columns
data.V1.plot(kind='line',color = 'm',label = 'V1',linewidth=1, alpha=0.5,grid=True,linestyle=':')
data.V2.plot(kind='line',color = 'g',label = 'V2',linewidth=1, alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper left')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('line plot')
plt.show()
data.plot(kind='scatter',x = 'V1',y= 'V2', alpha=0.5,color='r')
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('v1-v2 scatter plot')
plt.show()

data.V1.plot(kind='hist',bins = 50,figsize = (15,15))
plt.show()
data.V1.plot(kind='hist',bins = 50,)
plt.clf()
series = data['V1']
print(type(series))
data_frame=data[['V1']]
print(type(data_frame))
x = data['V1']>1
data[x][0:5]
data[np.logical_and(data['V1']>1,data['V2']<0)]
data[(data['V1']<1)&(data['V2']>0)]
for index,value in data[['V1']][0:3].iterrows():
    print(index , ':' , value)

