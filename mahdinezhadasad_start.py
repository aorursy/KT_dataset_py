# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.info()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths=0.5 ,ax=ax)

plt.show()
data.head()
data.corr()
data.columns

data.Speed.plot(kind='line', color='g',label='Legendary',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')



data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')

plt.ylabel('Defense')

plt.title(' Attack Defense Scatter Plot')

data.Speed.plot(kind= 'hist', bins=60, figsize=(12,12))

plt.show()

dictionary={'spain':'madrid','usa':'los angeles'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain']='Barcelona'

print(dictionary)



dictionary['france']='Paris'



print(dictionary)



del dictionary['spain']

print(dictionary)



print('france'in dictionary )



dictionary.clear()

print(dictionary)
x=data['Defense']>200



print(x)



data[x]
data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
lis=[1,3,2,5,35]

for i in lis:

    

    print('i in is',i)

    

print('')



for index, value in enumerate (lis):

    

    print(index," : " , value)

    

    

print('')



dictionary={'spain':'Madrid',' france':'Paris'}



for keys,values in dictionary.items():

    

    print(keys,' : ', values)

    

    

for key, value in data[['Attack']][0:5].iterrows():

    

    print(index,':', value)