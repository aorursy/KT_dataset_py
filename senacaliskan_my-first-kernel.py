# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/iris/Iris.csv")

data.head()
data.info()
data.corr()
f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = '.1f', ax=ax)

plt.show()
data.head(12)
data.columns
#linewidth = width of line, alpha = opacity, linestyle = sytle of line

data.SepalWidthCm.plot(color='r', label='SepalWidthCm', linewidth=1, alpha=0.5, grid=True, 

               linestyle='-')

data.SepalLengthCm.plot(kind='line', color='g', label='SepalLengthCm', linewidth=1, alpha=0.5, grid=True,

               linestyle=':')

plt.xlabel('x axis')      # the title of the x axis

plt.ylabel('y axis')      # the title of the y axis

plt.title('Line Plot')    # the title of the plot

plt.show()
#data.columns

#plt.scatter(data.SepalLengthCm, data.SepalWidthCm, color="red", alpha=0.5)

data.plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', alpha=0.5, color='red')

plt.xlabel('SepalLengthCm')

plt.ylabel('SepalWidthCm')

plt.title('SepalLengthCm SepalWidthCm Scatter Plot')

plt.show()
# bins = number of bar in figure

data.PetalLengthCm.plot(kind='hist', bins=50, figsize=(8,6))

plt.show()
# clf() = cleans it up again you can start a fresh

data.PetalLengthCm.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
# dictionary has 'key' and 'value'

# George is key, 24 is value

d = {"George": 24, "Tom": 32}

print(d.keys())

print(d.values())
d["Tom"] = 32

d["Jerry"] = 16
print(d["George"])
# keys are commonly strings or numbers

#d[10] = 100
d['George'] = "30"     # update

print(d)

d['Alice'] = "12"      # add new entry

print(d)

del d['George']     

print(d)

print('Alice' in d)
d.clear()   # remove all entries in d

print(d)
data = pd.read_csv("/kaggle/input/iris/Iris.csv")
series = data['SepalLengthCm']        # data['SepalLengthCm'] = series

print(type(series))

data_frame = data[['SepalLengthCm']]  # data[['SepalLengthCm']] = data frame

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1.Filtering Pandas data frame

x = data['SepalLengthCm']>7 
# 2.Filtering pandas with logical_and

# There are only 4 

data[np.logical_and(data['SepalLengthCm']>7, data['SepalWidthCm']>3 )]
# Stay in loop if condition (i is not equal 2) is true

i = 0

while i != 2 :

    print('i is: ',i)

    i +=1

print(i,' is equal to 2')
# Stay in loop if condition(i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

d = {'George':'24','Tom':'32'}

for key,value in d.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['SepalLengthCm']][0:1].iterrows():   # 0 is means the first element

    print(index," : ",value)
def tuple_ex():

    """ return defined t tuple"""

    t = (1,2,3)

    return t

a,b,c = tuple_ex()

print(a,b,c)
# access to index number



courses = ("Python Course", "C++ Course", "Java Course")



print(courses[1])
# Once a tuple has been created, you cannot change its values. The tuple cannot be changed.



courses = ("Python Course", "C++ Course", "Java Course")



courses[1]="C# Course"
# guess prints what

x = 2

def f():

    x = 3

    return x

print(x)      # x = 2 global scope

print(f())    # x = 3 local scope
# What if there is no local scope

x = 5

def f():

    y = 2*x        # there is no local scope x

    return y

print(f())         # it uses global scope x  

# First local scope searched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))

# what if we want to change default arguments

print(f(5,4,3))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)
# lambda function

square = lambda x: x**2     # where x is name of argument

print(square(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))
# Example of list comprehension

num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)