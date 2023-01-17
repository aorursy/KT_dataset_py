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
# Read data from file



data=pd.read_csv('/kaggle/input/fifa19/data.csv')
# Seeing the first 5 of the data

data.head()
# Getting information about data

data.info()
# Find out how many columns are in the data

data.columns
# Data correlation map

# Figsize: width and height values of the table to be formed

# Heatmap:table to be formed

# Cbar_kws: deciding whether the table is horizontal or vertical

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax,cbar_kws={"orientation": "horizontal"})

plt.show()

data.Age.plot(kind = 'line', color = 'g',label = 'Age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Special.plot(color = 'b',label = 'Special',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')

data.Overall.plot(color = 'r',label = 'Overall',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')



plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# Kind : The type of chart that will occur

# x = Composure, y = Marking

data.plot(kind='scatter', x='Composure', y='Marking',alpha = 0.3,color = 'blue')

plt.xlabel('Composure')              # label = name of label

plt.ylabel('Marking')

plt.title('Composure Marking Scatter Plot') 
# Histogram

# bins = number of bar in figure

data.Age.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
dictionarys={'ISTANBUL':'Eyüp','KASTAMONU':'Cide','IZMIR':'Bornova','ANTALYA':'Kemer'}



print(dictionarys.keys())

print(dictionarys.values())



# Nested Disitonarys 



myfamily = {

  "child1" : {

    "name" : "Emil",

    "year" : 2004

  },

  "child2" : {

    "name" : "Tobias",

    "year" : 2007

  },

  "child3" : {

    "name" : "Linus",

    "year" : 2011

  }

}



print(myfamily)
# update values

dictionarys['ISTANBUL']='Sisli'

print(dictionarys)



# add values

dictionarys['ANKARA']='Kecioren'

print(dictionarys)



#delete values

del dictionarys ['IZMIR']

print(dictionarys)



#check values

print('ANKARA' in dictionarys )



# clear data

dictionarys.clear()

print(dictionarys)



index=0

while index != 7 :

    

    print('index' , index)

    index=index+1

   

listdata=[1,2,3,4,5,6,7]



for each in listdata :

    print('index',each)

    

    # Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(listdata):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionarys={'ISTANBUL':'Eyüp','KASTAMONU':'Cide','IZMIR':'Bornova','ANTALYA':'Kemer'}

for key,values in dictionarys.items():

    print(key," : ",values)

print('')
''