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
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv") # we can read our dataset by read_csv function of pandas
# we can get general information from our dataset
data.info()
#for to see the columns of our dataset
data.columns
# with head() function, we can take a little look at our dataset to know what we are dealing with
data.head(10)
# similar function to head(), but we see the last 5 (or whatever you write in) samples of our dataset
data.tail(10)
# with corr() function, we see the correlation between the features of dataset
data.corr()
# correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.availability_365.plot(kind = "line",color = "y",label = "availability 365",alpha = 0.5,grid = True,linewidth = 1,linestyle = ":")
data.price.plot(color = "r",label = "price",alpha = 0.5,grid = True,linewidth = 1,linestyle = "-.")
plt.legend(loc = 'upper right') # legend = puts label into plot
plt.xlabel('x axis')            # label = name of label
plt.ylabel('y axis')
plt.title('availability - price')  # title = title of plot
plt.show()
# Scatter Plot 
# x = number of reviews, y = reviews per month

data.plot(kind = 'scatter',x = 'number_of_reviews',y = 'reviews_per_month',color = 'red',alpha = 0.5)
plt.xlabel('number of reviews')            # label = name of label
plt.ylabel('reviews_per_month')
plt.title('reviews')  # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.availability_365.plot(kind = 'hist',bins = 15,figsize = (7,7))
plt.show()
# clf() = cleans it up again you can start a fresh
data.availability_365.plot(kind = 'hist',bins = 50)
#plt.clf()
# We cannot see plot due to clf()
#create dictionary and look its keys and values
dictionary = {'turkey' : 'ankara','russia' : 'moscow'}
print(dictionary.keys())
print(dictionary.values())
#Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['turkey'] = "istanbul"    # update existing entry
print(dictionary)
dictionary['kenya'] = "nairobi"       # Add new entry
print(dictionary)
del dictionary['kenya']              # remove entry with key 'kenya'
print(dictionary)
print('turkey' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
# In order to run all code you need to take comment this line
del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# 1 - Filtering Pandas data frame
x = data['price']>8000    # There are only 7 airbnb rooms who have higher price value than 8000
data[x]
# 2 - Filtering pandas with logical_and
# There are only 2 airbnb rooms who have higher price value than 8000 and higher minimum nights value than 50
data[np.logical_and(data['price']>8000, data['minimum_nights']>50 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['price']>8000) & (data['minimum_nights']>50)]
def tuble_ex():
    """ return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)
#nested function
def square():
    """ return square of value """
    def add():
        """ add two local variable """
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2
print(square())    
#Default argument example:
def f(a, b=1):
  """ b = 1 is default argument"""
#Flexible argument example:
def f(*args):
 """ *args can be one or more"""

def f(** kwargs):
 """ **kwargs is a dictionary"""

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
# Anonymus function
number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))
# iteration example
name = "ronaldo"
it = iter(name)
print(next(it))    # print next iteration
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
# Example of list comprehension
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)
# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.
threshold = sum(data.price)/len(data.price)
print("threshold",threshold)
data["price_level"] = ["high" if i > threshold else "low" for i in data.price]
data.loc[:10,["price_level","price"]] # we will learn loc more detailed later
data.describe()