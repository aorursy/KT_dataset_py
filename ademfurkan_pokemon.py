# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualization tools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'orange',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'blue',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
#Scatter Plot

# x = attack , y= defense

data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')              # label = name of label

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Speed.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
dictionary = {"Spain":"Madrid","USA":"Vegas"}

print(dictionary.keys())

print(dictionary.values())
#Add new keys and values

dictionary['France'] = "Paris"              #{'Spain': 'Madrid', 'USA': 'Vegas', 'France': 'Paris'}

print(dictionary)

#Update keys and values

dictionary['Spain'] = "Barcelona"           #{'Spain': 'Barcelona', 'USA': 'Vegas', 'France': 'Paris'}

print(dictionary)

#Remove keys

del dictionary['Spain']                     #{'USA': 'Vegas', 'France': 'Paris'}

print(dictionary)

#Checking

print('France' in dictionary)               #True

print('England' in dictionary)               #False

#Clearing

dictionary.clear()

print(dictionary)                           #{}
#İf you write -del dictionary- the program will give error.

#del dictionary ////(Error is "NameError: name 'dictionary' is not defined")

print(dictionary)
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
#İf you use command "Defense" for Series -[Defense]- for DataFrame -[[Defense]]-

series = data['Defense']        # data['Defense'] = series

print(type(series))

data_frame = data[['Defense']]  # data[['Defense']] = data frame

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
#Filtering

x = data['Defense'] > 200  #We have Only 3 pokemon defense > 200

data[x]
#2 Filtering

data[np.logical_and(data["Defense"]>200,data["Attack"]>100)]
#2 Filtering(Other methots)

data[(data["Defense"]>200) & (data["Attack"]>100)]
i = 0

while i != 5:

    print("i is" , i)

    i+=1

print("i is equal to 5")
list1 = [1,2,3,4,5]

for i in list1:

    print("i is",i)

print('')

# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(list1):

    print(index," : ",value)

print('')

# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['Attack']][0:1].iterrows():

    print(index," : ",value)