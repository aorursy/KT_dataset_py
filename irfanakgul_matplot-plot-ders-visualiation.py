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
data = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')

data.info()
data.corr()


f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')              # label = name of label

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.Speed.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Speed.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['spain'] = "barcelona"    # update existing entry

print(dictionary)

dictionary['france'] = "paris"       # Add new entry

print(dictionary)

del dictionary['spain']              # remove entry with key 'spain'

print(dictionary)

print('france' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
# In order to run all code you need to take comment this line

del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/pokemon/Pokemon.csv')

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
# 1 - Filtering Pandas data frame

x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200

print(x)

data[x]
 #2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['Defense']>200, data['Attack']>100 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['Defense']>200) & (data['Attack']>100)]
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

for index,value in data[['Attack']][0:1].iterrows():

    print(index," : ",value)
num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

for i in num1:

    if i == 10 :

        print(i**2)

    else :

        print(i-5)

num3 = [i**2 if i < 7 else i+5 for i in num1]

print(num2)

print(num3)
def tuble_ex():

    """ return defined t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# guess print what

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

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
import builtins

dir(builtins)
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

print(un_zip)

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
 #lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later
data.info()
data.describe()
# For example lets look frequency of pokemom types

print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
# For example lets look frequency of pokemom types

print(data['Type 2'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
# For example max HP is 255 or min defense is 5

data.describe() #ignore null entries
# For example max HP is 255 or min defense is 5

data['Type 2'].describe() #ignore null entries
data.dropna(inplace = True)  

data.describe()
# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='Attack',by = 'Legendary')
data_new = data.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Name', columns = 'variable',values='value')
# Firstly lets create 2 data frame

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data3 = data['Name'].head()

data1 = data['Attack'].head()

data2= data['Defense'].head()

conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes

# lets convert object(str) to categorical and int to float.

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')
data.info() 

data.head(33)
# Lets chech Type 2

data["Type 2"].value_counts(dropna =False) 

# As you can see, there are 386 NAN value
# Lets drop nan values

data1=data   # also we will use data to fill missing value so I assign it to data1 variable

data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?



data1=data

data1.dropna(subset=['Type 2'],inplace=True)

data1.head(30)

# type 2 deki tum Nan lari silecek
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true



# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false

assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
data["Type 2"].fillna('empty',inplace = True)



assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

assert data.columns[1] == 'Name'

assert data.Speed.dtypes == np.int





#bu kodlarla sutunlarin typelerini tek seferde sorgulayabiliriz, 

# data frames from dictionary

country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
#yeni kolon ekleme



df["capital"] = ["madrid","paris"]

df
# Broadcasting

df["income"] = 0 #Broadcasting entire column

df
# Plotting all data 

data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()

time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
# Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])