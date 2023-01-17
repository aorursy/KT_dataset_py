# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
data=pd.read_csv('../input/tmdb_5000_movies.csv')
data.info()
data.columns
data.head(10)
averageRuntime=data.runtime.mean()

print(averageRuntime)
data2 = data.tail()

print(data2)
data1 = data.runtime.head()

data2 = data.runtime.tail()



data_v_concat= pd.concat([data1,data2],axis = 0)

data_h_concat = pd. concat([data1,data2],axis =1)

print(data_v_concat)

print(' ')

print(data_h_concat)
data.corr()
sns.heatmap(data.corr())

f,ax=plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(),annot=True, linewidth=8,fmt='.2f',ax=ax)



data.budget.plot(kind='line',color='r', label='budget',linewidth=1,alpha=1,grid=True,linestyle=':')

data.runtime.plot(kind='line',color='y',label='runtime',linewidth=1,alpha=0.5,grid=True,linestyle=':')

plt.legend(loc='upper right')

plt.xlabel=('budget')

plt.ylabel=('revenue')

plt.title('Line Plot')
data.plot(kind='scatter',x='runtime',y='popularity',alpha=1,color='pink' )

plt.xlabel=('runtime')

plt.ylabel=('popularity')
data.popularity.plot(kind='hist',bins=20,figsize=(18,6))

plt.show()
Dictionary={'Footname':'Sushi','Drinkname':'Coffee','Dessertname':'Cheesecake'}

print(Dictionary.keys())

print(Dictionary.values())
Dictionary['Dessertname']="Tirasmisu" ##for update
print(Dictionary.values())
Dictionary['ColdDrinkname']="Lemonada" ##Add new entry
del Dictionary['Footname'] ##delete Footname data
del Dictionary ##delete dictionary for free up memory
Series=data['runtime']

print(type(Series))

dataFrame=data[['runtime']]

print(type(dataFrame))
data.info()
x=data['runtime']>300

data[x]
data[np.logical_and(data['revenue']>800000,data['runtime']>250)]

i=0

while i!=10:

    print(i)

    i+=1
list1=[1,2,3,4,5]

for i in list1:

    print(i)

    
dictionary={'trial1':'one','trial2' : 'two'} #for index and value

for  key,value in dictionary.items():

    print(key,":",value)



for index,i in enumerate(list1): #for index : number

    print(index ,":",i)
for  index,i in data[['runtime']][0:1].iterrows():

    print(index ,"",i)