# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/youtube-new/CAvideos.csv')
data.info
data.corr()
#correlation map

f,ax=plt.subplots()

sns.heatmap(data.corr(),annot=True,linewidths=.2,fmt='.2f',ax=ax )
data.head(10)
data.colums
for col in data.columns:

    print(col)

    

    #veya 

    

    list(data.columns.values)
data.views.plot(kind='line', color='b', label='view', linewidth=1, alpha=0.5, grid=True,linestyle=':')

data.likes.plot(color='r', label='likes', linewidth=1, alpha=1, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Lİne Plot')
data.plot(kind='scatter', x='views', y='likes',alpha=0.5,color= 'red')

plt.xlabel('views')

plt.ylabel('likes')

plt.title('ScatterPlot')
#veya aynı grafik aşağıdaki gibi de oluşturulabilir,

plt.scatter(data.views, data.likes, color="red", alpha=0.5)
data.plot(kind='bar',x='views',y='likes')
data.views.plot(kind='hist',bins=50,figsize=(10,10))

plt.show()
data.views.plot(kind='hist',bins=50,figsize=(10,10))

plt.clf()
series=data['likes'] #series tek köşeli parantez seri

print(type(series))

data_frame=data[['likes']] #çift köşeli parantez frame

print(type(data_frame))
x=data['likes']>100000

data[x]
data[np.logical_and(data['likes']>100000,data['views']>100000)]
#veya bu şekilde de yapılabilir

data[(data['likes']>100000) & (data['views']>100000)]
for index.value in data[['title']][0:1].iterrows():

    print(index,":",value)
threshold= sum(data.likes)/len(data.likes)

print("threshold",threshold)

data["like_level"]=["high" if i>threshold else "low" for i in data.likes]

data.loc[:10,["like_level","likes"]]