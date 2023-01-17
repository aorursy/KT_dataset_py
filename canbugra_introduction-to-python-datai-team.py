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



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/master.csv")

data.info()
data.corr()
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot = True, linewidths = .5,fmt = '.1f',ax = ax)

plt.show()
data.head()
data.head(10)
data.columns
#Line Plot

data.population.plot(kind = 'line',color = 'g',label = 'population',linewidth=10,alpha = 0.5,

                            grid = True,linestyle = ':')

data.year.plot(kind = 'line',color = 'r',label = 'year',linewidth=20,alpha=0.5,

                      grid =  True,linestyle = '-.')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
#Scatter Plot

data.plot(kind = 'scatter',x = 'population',y = 'year',alpha = 0.5, color = 'red')

plt.xlabel('population')

plt.ylabel('year')

plt.title('Population versus Year')

plt.show()
#Histogram

data.population.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.population.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()

#create dictionary and look its keys and values

dictionary = {"Name":["ali","veli","kenan","hilal","ayse","evren"],

              "Age":[15,16,17,33,45,66],

              "Maas":[100,150,240,350,110,220]}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary = {"Name":["ali","veli","kenan","hilal","ayse","evren"],

              "Age":[15,16,17,33,45,66],

              "Maas":[100,150,240,350,110,220]}

list1 = ["ali","veli","kenan","hilal","ayse","evren"]

#Reverse the list

#Then attend list to dictionary

list2 = list1[::-1]

dictionary['Name'] = list2

print(dictionary)



list1 = ["ali","veli","kenan","hilal","ayse","evren"]

list1.reverse()



dictionary['Name'] = list1

print(dictionary)



    

list2 = [15,16,17,33,45,66]

list2.sort()

dictionary['Age'] = list2

print(dictionary)

    

    
dictionary.clear()                   # remove all entries in dict

print(dictionary)

data = pd.read_csv("../input/master.csv")

data.info()
series = data['country']       

print(type(series))

data_frame = data[['country']]  

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['population'] > 20000     

data[x]

# 2 - Filtering pandas with logical_and



data[np.logical_and(data['population']>20000, data['suicides_no']>10000 )]
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['suicides_no']][0:5].iterrows():

    print(index," : ",value)