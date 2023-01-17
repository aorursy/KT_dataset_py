# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/CAvideos.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.category_id.plot(kind = 'line', color = 'g',label = 'category_id',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.views.plot(color = 'r',label = 'views',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')



plt.legend(loc='upper right')  



plt.title('Line Plot')



plt.show()
data.plot(kind='scatter',x = 'category_id',y = 'views',color='red')

plt.show()
data.category_id.plot(kind='hist',bins=50,color='red',figsize=(12,12))

plt.show()
dictionary = {'barcelona':'messi','juventus':'ronaldo','fenerbahce':'Alex de Souza'}

print(dictionary.keys())

print(dictionary.values())
series = data['views']

dataframe = data[['category_id']]

print(type(series))

print(type(dataframe))
k = data[(data['category_id']>10) & (data['views'] < 100000)]
data[k]
data[np.logical_and(data['views']<10000,data['category_id']>10)]
for key,value in dictionary.items():

    print(key,":",value)

    

print()



for index,value in data[['views']][0:5].iterrows():

    print(index,":",value)
data.boxplot(column='views',by='category_id')