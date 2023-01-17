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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='.2f',ax=ax)
plt.show()
data.columns
data.head()
data.head(10)
data.runtime.plot(kind='line',color='red',label='runtime',linewidth=2,alpha=0.5,grid=False,linestyle=':')
plt.legend(loc='upper right') 
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.title('Line')
plt.show()

data.plot(kind='scatter',x='budget',y='popularity',alpha=0.5,color='green')
plt.xlabel('budget')
plt.ylabel('popularity')
plt.title('Scatter Budget-Popularity')
plt.show()
plt.scatter(data.budget,data.popularity,color='green',alpha=0.5)
plt.show()
data.budget.plot(kind='hist',bins=20,figsize=(20,10))
data.budget.plot(kind='hist',bins=20)
plt.clf()
x=data['original_language']!='en'
data[x]
data[np.logical_and(data['popularity']>300, data['revenue']>100 )]

data[(data['revenue']>100)&(data['popularity']<200)]
series=data['original_title']
print(type(series))
data_frame= data[['tagline']]
print(type(data_frame))
for index,value in data[['popularity']][0:1].iterrows():
    print(index,":",value)
print('King Kong' in data.original_title)

print('original_title'in data)
print(data.original_language.unique())