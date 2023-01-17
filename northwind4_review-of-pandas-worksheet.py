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
country = ['Turkey', 'Germany']#values

population = ['80000000', '65000000']#values

list_label = ['country', 'population']#column names

list_column = [country, population]#values have relation with columns 

zipped = list(zip(list_label,list_column))#mwe used zip() method for creating a table form

data_dict = dict(zipped)#zip() method returns tuple data type. We have to change it to dictionary for using with Pandas

data = pd.DataFrame(data_dict)#dictionary transformed into Data Frame.

data
data['capital'] = ['Ankara', 'Berlin']

data
data['test'] = '123'

data
data1 = pd.read_csv('../input/2017.csv') #creating new data frame from World Happiness Report dataset.

data1.head()

data1_cols = data1.columns

data1_cols = data1_cols.str.replace('.','_')

data1.columns = data1_cols

data1.head()
data2 = data1.loc[:,['Happiness_Rank', 'Family', 'Freedom']]

data2.plot()
data2.plot(subplots = True, figsize= (12,12))

plt.show()
data2.plot(kind='scatter', x= 'Happiness_Rank', y='Freedom', color = 'Red',grid = True, alpha=0.5, figsize = (12,12))

plt.show()
data2.plot(kind='hist', y= 'Happiness_Rank', bins = 50, range=(0,250), density=True)

plt.show()
fig,axes = plt.subplots(nrows=2 , ncols=1)

data2.plot(kind='hist', y='Freedom', bins = 50, range=(0,1), density=True, ax= axes[0])

data2.plot(kind='hist', y='Happiness_Rank', bins= 50, range=(0,250), density=True, ax=axes[1], cumulative = True)

plt.savefig('graph.png')

plt.show()
data1.describe()
time_list=["1995-04-23","1999-11-14","1989-1-17","1996-6-5","1999-11-4"]

print(type(time_list[1]))#returns string

# lets convert it to datetime object

data3 = data2.head()

datetime_object = pd.to_datetime(time_list)#converting process to datetime

data3['date'] = datetime_object#adding to dataframe as a column

data3
data3 = data3.set_index('date')

data3
print(data3.loc['1999-11-14'])

print(data3.loc['1989-01-17':'1999-11-04'])
data3.resample('A').mean()
data3.resample('M').mean()
data3.resample('M').first().interpolate('linear')
data3.resample("M").mean().interpolate("linear")