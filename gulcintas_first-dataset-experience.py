# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')
data.info()
data.head()
data.shape
data.columns
data.dtypes
data.describe()
# the most 5 attacker pokemon is 

data.sort_values(by="Attack" ,ascending = False ).head()
#correlation map

f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(data.corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()
data.head(2)
# line plot

data.Attack.plot(kind='line', color='blue' ,label = 'Attack',linewidth=1,alpha=0.5 ,grid = True,

                linestyle = ":")

data.Speed.plot(kind ='line', color='red', label ='Speed',linewidth=1,alpha=0.5,grid=True,

               linestyle = "-.")
# Scatter Plot 

# x = Speed, y = Attack

data.plot(kind='scatter', x='Speed', y='Attack',alpha = 0.5,color = 'red')

plt.xlabel('Speed')              # label = name of label

plt.ylabel('Attack')

plt.title('Speed Attack Scatter Plot')     
# histogram

data.Attack.plot(kind='hist',bins=50,figsize=(12,12))

plt.show()
# Let's create a dictionary and look it's values and keys

dictionary = {'1':'Istanbul','2':'Izmır','3':'Ankara','4':'London','5':'Boston'}

print(dictionary.keys())

print(dictionary.values())
dictionary['1']='Bursa'          # update

dictionary['6']='İstanbul'       # adding new key value

print(dictionary) 

del dictionary['3']              # remove entry with key 3

print(dictionary)

print('3' in dictionary)         # check include or not
dictionary.clear()               # remove all entries in dict

print(dictionary)

# pandas

data = pd.read_csv('../input/pokemon.csv')
series = data['Speed']

print(type(series))

data_frame = data[['Speed']]

print(type(data_frame))
# filtering data with Pandas data frame

x = data['Speed']>150

data[x]
# filtering data pandas with logical and

data[np.logical_and(data['Speed']>145,data['Attack']>100)]
data[(data['Speed']>145)&(data['Attack']>100)]

lis = [1,2,3,4,5]

for i in lis:

    print('i is :',i)

print("")

    

for index,value in enumerate(lis):

    print(index,":",value)

    

dictionary = {'Turkey':'Ankara','England':'Londra'}

for key, value in dictionary.items():

    print(key,':',value)

    print('')

    

for index,value in data[['Speed']][0:1].iterrows():

    print(index,":",value)