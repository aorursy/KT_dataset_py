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
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.info()
df.corr()
df.head(20)
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

df.Confirmed.plot(kind = 'line', color = 'g',label = 'Confirmed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.Deaths.plot(color = 'r',label = 'Deaths',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# example of what we learn above

def tuble_ex():

    t = (3,5,7,8)

    return t

a,b,c,d = tuble_ex()

print(a,b,c,d)
x = 2

def f():

    x = 3

    return x

print(x)      # x = 2 global scope

print(f())    # x = 3 local scope
x = 10

def f():

    y = 2*x        # there is no local scope x

    return y

print(f()) 
import builtins

dir(builtins)
def resullt():

    """ return square of value """

    def add():

        """ add two local variable """

        x = 4

        y = 5

        z = x + y

        return z

    return add()**3

print((resullt()))    
# default arguments

def f(a, b = 3, c = 5):

    y = a + b + c

    return y

print(f(5))

# what if we want to change default arguments

print(f(5,4,3))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(3)

print("")

f(1,2,3,4,5,6,7)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 654321)
total = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(total(12,45,78))
number_list = [6,3,8]

y = map(lambda x:x**2,number_list)

print(list(y))
# iteration example

city = "İSTANBUL"

it = iter(city)

print(next(it))

print(next(it))

print(*it)         # print remaining iteration

# zip example

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)



un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list2))
num1 = [10,20,30,40,50,60,70,80,90,40]

num2 = ["geçti" if i > 45 else "kaldi"  for i in num1]

print(num2)
x = df["Confirmed"].head(30)

confirmed = ["risky area" if i > 20 else "non-risky area" for i in x]

print(confirmed)