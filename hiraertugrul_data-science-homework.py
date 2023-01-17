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
data = pd.read_csv('../input/world-happiness/2015.csv')

data.info
data.corr()
f, ax = plt.subplots(figsize = (12, 12))

sns.heatmap(data.corr(), annot= True, linewidths= .8, fmt= '.1f', ax= ax)

plt.show()

data.head(5)
data.columns
f, ax = plt.subplots(figsize = (20, 7))

data['Happiness Score'].plot(kind= 'line', color= 'g', label= 'Happines Score', linewidth= 3, alpha= .5, linestyle= '-')

data['Economy (GDP per Capita)'].plot(kind= 'line', color= 'r', label= 'Economy (GDP per Capita)', linewidth= 2, alpha= .5, linestyle= '-')

plt.legend(loc = 'upper right')

plt.xlabel('Countries')

plt.ylabel('Values')

plt.title('Line Chart')

plt.show()
data.plot(kind= 'scatter', x='Happiness Score', y= 'Economy (GDP per Capita)')

data.plot(kind= 'scatter', x='Family', y= 'Economy (GDP per Capita)', color= 'm')

data.plot(kind= 'scatter', x='Happiness Rank', y= 'Economy (GDP per Capita)', color= 'c')

plt.show()
data['Happiness Score'].plot(kind= 'hist', bins= 50, color= 'y', figsize = (10,7))

plt.show()
dictionary = {'name' : 'alex','surname' : 'vargas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['name'] = "camilla"    # update existing entry

print(dictionary)

dictionary['age'] = 12       # Add new entry

print(dictionary)

del dictionary['name']              # remove entry with key 'spain'

print(dictionary)

print('surname' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
del dictionary

print(dictionary)  
data = pd.read_csv('../input/world-happiness/2015.csv')

series = data['Freedom']      

print(type(series))

data_frame = data[['Generosity']] 

print(type(data_frame))
print(5 < 2)

print(3==2)

print(True and False)

print(True or False)
x = data['Trust (Government Corruption)']>0.5

data[x]
data[np.logical_and(data['Economy (GDP per Capita)']<1, data['Happiness Score']>7 )]
data[(data['Economy (GDP per Capita)']<1) & (data['Happiness Score']>7)]
t= 0

while t!=5:

    print(t)

    t=t+1

    
my_list= ['a', 'b','c','d' ]



for i in my_list:

    print("i=",i)

for i,value in enumerate(my_list):

    print("index:",i,",value:", value)
dictionary = {'spain':'madrid','france':'paris'}



for key,value in dictionary.items():

    print('key:',key,',value:',value)
for index,value in data[['Happiness Score']][0:2].iterrows():

    print(value)