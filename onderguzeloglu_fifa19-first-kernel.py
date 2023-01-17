# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns       # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read csv file

data = pd.read_csv('../input/fifa19/data.csv')
data.info()
data.columns
# show top 10 sample

data.head(10)
# show features correlation

data.corr()
# Correlation map

f,ax = plt.subplots(figsize =(18,18))

sns.heatmap(data.corr(), annot = True, linewidths =.5, fmt = '.1f', ax =ax)

plt.show()
# Line plot of Potential and Overall

data.Potential.plot(kind = 'line',color = 'c', label = 'Potential', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')

data.Overall.plot(color = 'r', label = 'Overall', linewidth = 1, alpha = 0.5, grid = True, linestyle = '--')

plt.legend(loc = 'upper right')      # legend = puts label into plot

plt.xlabel('x axis')                 # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')               # title title of plot

plt.show()
# Scatter Plot of Shot Power and Finishing

# x = ShotPower, y = Finishing



data.plot(kind = 'scatter', x = 'ShotPower',y = 'Finishing', alpha = 0.5, color = 'red')

plt.xlabel('shotPower')         # label = name of label

plt.ylabel('Finishing')

plt.title('Shot Power and Finishing Scatter Plot')

plt.show()
#Histogram of finishing

#bins = number of bar in figure

plt.title('Histogram of Finishing')

data.Finishing.plot(kind = 'hist', bins = 50, figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Finishing.plot(kind = 'hist',bins = 50)

plt.clf()

#We can't see plot due to clf()
# Create dictionary and look its keys and values



dictionary = {'Spain':'Madrid','USA':'Vegas'}

print(dictionary.keys())

print(dictionary.values())
#Keys have to immutable objects like string,boolean,float,integer or tubles

#list is not immutable

#keys are unique



dictionary['Spain'] = "Barcelona"  #update existing entry

print(dictionary)

print('')

dictionary['France'] = "Paris"     # add new entry

print(dictionary)

print('')

dictionary['Year'] = 2021

print(dictionary)

print('')

for x,y in dictionary.items():      # a different way to write keys and values

    print(x, '-', y)

print('')

del dictionary['Spain']             # Remove entry with key 'Spain'

print(dictionary)

print('')

print('France' in dictionary)       # check include or not

dictionary.clear()

print(dictionary)
# In order run all code youe need to take comment this line

# del dictionary     #delete entire dictionary

print(dictionary)
data = pd.read_csv('../input/fifa19/data.csv')
series = data['Name']      # data['Name']

print(type(series))

data_frame = data[['Name']]     #data[['Name']] = data_frame

print(type(data_frame))
#Comparison operator

print(3 > 2)

print(3 != 2)

#Boolean operators

print(True and False)

print(True or False)
# 1- Filtering Pandas data frame



x = data['Potential'] > 90  # there are 29 football player who have higher potential value than 90

data[x]
# 2- Filtering Pandas data frame

# There are 9 football players who have higher potential value than 90 Overall value higher than 90



data[np.logical_and(data['Potential']>90, data['Overall']>90)]
#this is also same with,previous code line. therefore we can also use '&' for filtering.

data[(data['Potential']>90) & (data['Overall']>90)]
# Stay in loop if condition(i is not equal 5) is true

i = 0

while i != 5:

    print('i is: ',i)

    i += 1

print(i, 'i is equal to 5')
# Stay in loop if condition (i is not 5) is true

list = [1,2,3,4,5]

for i in list:

    print('i is: ',i)

print('')



#ennumerate index and value of list

#index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(list):

    print(index, " : ", value)

print('')





# For dictionaries

# We can use for loop to achive key and value of dictionary. We learned key and value at dictionary part.

dictionary = {'Spain':'Barcelona','France':'PSG'}

for key,value in dictionary.items():

    print(key, ' : ', value)

print('')





#For pandas we can achieve key and value



for index, value in data[['Potential']][0:1].iterrows():

    print(index, ' : ', value)
# docstrings

def tuple_ex():

    """return defined t tuples"""

    t = (1,2,3)

    return t

a,b,c = tuple_ex()

print(a,b,c)

    
# guess print what

row1 = data[['Name']][0:1]



def f():

    row1 = data[['Name']][1:2]

    return row1

print(row1)        #row1 = data[['Name']][0:1] global scope

print(f())         # row1 = data[['Name']][1:2] local scope 

        
# what if there is no local scope

x = 5



def f():

    y = 2 * x      # there is no local scope x

    return y   

print(f())         #  it uses global scope x

# First local scope searched, then global scope searched, if to of them cannot be found lastly built in scope searched
#How can we learn what is built in scope



import builtins

dir(builtins)
# nested function

def square():

    """ return square of value """

    def add():

        """ add two local variable """

        x = 2

        y = 5

        z = x + y

        return z 

    return add()**2 

print(square())
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))



# what if we want to change default arguments

print(f(5,4,3))
# flexible arguments *args



def f(*args) :

    for i in args:

        print(i)

f(0)

print("")

f(1,2,3,4)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():

        print(key, " ", value)

f(Name = data[['Name']][:2], Club = data[['Club']][:2])
# lambda function

square = lambda x : x**2       # where x is name of argument

print(square(4))

tot = lambda x,y,z: x+y+z      # where x,y,z are names of arguments

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(y)
#Iteration example

name = data[['Name']][0:1].items()

it = iter(name)

print(next(it))      # print next iteration

print(*it)           # print remaining iteration
#Zipping list

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
#Unzipping

un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip)       # unzip return tuple

print(un_list1)

print(un_list2)

print(type(un_list2))
# Example of list comprehension

num1 = [1,2,3]

num2 = [i + 1 for i in num1]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
# list comprehension example with our dataset

threshold = sum(data.Potential)/len(data.Potential)

data["potential_Level"] = ["Good_Player" if i > threshold else "Bad_Player" for i in data.Potential]

data.loc[12000:15000,["potential_Level","Potential"]]