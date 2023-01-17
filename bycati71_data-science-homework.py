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

import codecs

    
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/NBA_player_of_the_week.csv')

data.info()

data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(1000)

data.columns

# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data['Seasons in league'].plot(kind = 'line', color = 'g',label = 'Seasons in league',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Age.plot(color = 'r',label = 'Age',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Seasons in league', y='Age',alpha = 0.5,color = 'red')
plt.xlabel('Real_value')
plt.ylabel('Age')
plt.title('Value Date Scatter Plot')
# Histogram
# bins = number of bar in figure
data.Age.plot(kind = 'hist',bins = 40,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data['Seasons in league'].plot(kind = 'hist',bins = 50)
plt.clf()
#create dictionary and look its keys and values
dictionary = {'Spain' : 'Barcelona',
              'Turkey' : 'Fenerbahce',
              'Germany' : 'Bayern Munih',
              'France' : 'Paris Saint Germain'}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['spain'] = "barcelona"
print(dictionary)
dictionary['france'] = "paris"
print(dictionary)
del dictionary['spain']
del dictionary['france']
print(dictionary.keys())
print(dictionary.values())
print(dictionary)
print('france' in dictionary)
print('France' in dictionary)
dictionary.clear()
print(dictionary)
series = data['Team']
print(type(series))
data_frame = data[['Team']]
print(type(data_frame))
# Comparison operator
print(6 > 9)
print(70 >= 15)
print(4!=7)
# Boolean operators
print(True and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data['Age']<30
data[x]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['Age']<25) & (data['Real_value']<1)]
# Stay in loop if condition( i is not equal 18) is true
i = 0
while i != 18 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 18')
# Stay in loop if condition( i is not equal 9) is true
liste = [1,2,3,4,5,6,7,8,9]
for i in liste:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(liste):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'Spain':'Real Madrid','France':'Paris Saint Germain', 'Italy' : 'Juventus'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['Age']][0:1].iterrows():
    print(index," : ",value)
def tuble_ex():
    t = (data.Team, data.Player, data.Age)
    return t
a,b,c = tuble_ex()
print(a)
print(b)
print(c)
sL = data['Seasons in league']
def f():
    x = data.Real_value
    y = sL / x
    return y
print(f())
import builtins
dir(builtins)
dir(pd)
dir(plt)
def square():
    
    def add():
        sL = data['Seasons in league']
        x = data.Real_value
        y = sL / x
        return y
    return add()**2
print(square())
def f(**kwargs):
    for key, value in kwargs.items():
        print(key, " ", value)
f(Team = data.Team, Player = data.Player, Age = data.Age)

sL = data['Seasons in league']
x = data.Age
square = lambda z,y: y/z
print(square(sL,x))

sL = data['Seasons in league']
rv = data.Real_value
y = map(lambda x:x**rv,sL)
print(list(y))
# iteration example
name = data.Player
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration
list1 = data.Player
list2 = data.Team
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))
# Example of list comprehension
num1 = data['Seasons in league']
num2 = [i * 2 for i in num1 ]
print(num2)
num1 = data['Seasons in league']
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
threshold = sum(data.Real_value)/len(data.Real_value)
data["speed_level"] = ["high" if i > threshold else "low" for i in data.Real_value]
data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later
