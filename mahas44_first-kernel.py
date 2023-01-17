# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/tmdb_5000_movies.csv")
data.info()
data_cor = data.corr()

data_cor
# correlation map



f,ax = plt.subplots(figsize =(18,18))

sns.heatmap(data_cor, annot = True, linewidths = 5, fmt = '.3f', ax = ax)

plt.show()





# data.head(15) # We don't need the other features



title = data[['original_title','budget','revenue', 'genres']]

title.head(15)
data.columns
# Line Plot



data.revenue.plot(kind = 'line', color = 'b', label = 'revenue', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')

data.budget.plot(color = 'r', label = 'budget', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')

plt.legend()

plt.xlabel('Movie Count')

plt.ylabel('Money(m)')

plt.title('Line Plot')

plt.show()

# Scatter Plot



data.plot(kind='scatter', x = 'budget', y = 'revenue', alpha = 0.3, color = 'blue')

plt.xlabel('Budget')

plt.ylabel('Revenue')

plt.title('Budget-Revenue Scatter Plot')

plt.show()
# Histogram

# bins = number of bar in figure



data.budget.plot(kind = 'hist', bins = 50 , figsize = (12,12))

plt.xlabel('Budget')

plt.show

# clf() = clean it up again you can start a fresh



data.budget.plot(kind = 'hist', bins = 50)

plt.clf()



# We can't see plot due to clf()
# Create dictionary and look its keys and values



dictionary = {'Avatar': 'Action', 'Pirates of Carabian V': 'Adventure'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique



dictionary['Avatar'] = "Fantastic"         # Updated "Avatar" value

print(dictionary)



dictionary['Tangled'] = "Animation"        # Added new object

print(dictionary)



del dictionary['Pirates of Carabian V']    # Deleted "Pirates of Carabian V"

print(dictionary)



print('Avatar' in dictionary)              # Check 'Avatar' include or not (return True or False)

dictionary.clear()                         # remove all entries in dict

print(dictionary)

# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv('../input/tmdb_5000_movies.csv')
series = data['budget']    # data['budget'] = series

print(type(series))



data_frame = data[['budget']]  # data[['budget']] = data frame

print(type(data_frame))
# Filtering Pandas data frame



x = data['budget'] > 250000000

data[x]
data[np.logical_and(data['budget'] > 250000000, data['revenue'] > 400000000)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.



data[(data['budget'] > 250000000) & (data['revenue'] > 400000000)]
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

dictionary = {'Avatar': 'Action', 'Pirates of Carabian V': 'Adventure'}

for key, value in dictionary.items():

    print(key," : ", value)

print('')



# For pandas we can achieve index and value

for index,value in data[['budget']][0:2].iterrows():

    print(index, " : " , value)
# User defined function

# example of what we learn above



def tuble_ex():

    t = (1,2,3)

    return t



a,b,c = tuble_ex()

print(a,b,c)
# Scope



# global : defined main body in script

# local : defined in a function

# built in scope : names in predefined built in scope module such as print, len



x = 4

def f():

    x = 1

    return x



print(x)  # x = 4 global scope

print(f()) # x = 2 local scope
# What if there is no local scope



x = 3

def f():

    y = 3*x  # There is no local scope x

    return y

print(f()) # it uses global scope x



# First loacal scope searched, then global scope searched,

# if two of them cannot be found lastly built in scope searched
import builtins

dir(builtins)
# NESTED FUNCTION



def square():

    # return square of value

    def add():

        # add two local variable

        

        x = 3

        y = 5

        z = x + y

        return z

    return add()**2

print(square())

# Default Arguments



def f(a, b = 1, c = 2): # if  only 1 parameter is given, b = 1, c = 2 accept by function

    y = a + b + c

    return y

print(f(6)) # a = 6

# what if we want to change default parameter

print(f(5,4,6))

#Flexible Arguments

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)



# flexible arguments **kwargs that is dictionary



def f(**args):

    # print key-value of dictionary

    for key,value in args.items():

        print(key," ", value)

        

f(title = 'Avatar', genre = 'Action', budget = 12300000)

# LAMBDA FUNCTION



# user defined function (long way)

def square(x):

    return x**2

print(square(5))



# lambda function (short way)



square = lambda x : x**2       # where x is name of argument

print(square(3))



tot = lambda x,y,z: x*y*z      # wehre x,y,z are names of arguments

print(tot(3,4,5))
# Anonymous Function

# map(func,seq) : applies a function to all them items in a list

number_list = [1,2,3]

y = map (lambda x:x**2,number_list)

print(list(y))
# Iterator example

# iterable is an object that can return an iterator

# iterable: an object with an associated iter() method; ex: list,strings and dictionaries

# iterator: produces next value with next() method



name = "Avatar"

it = iter(name)

print(next(it))   # print next iteration

print(*it)        # print remaining iteration
# zip() : zip list



list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)



z_list = list(z)

print(z_list)



print(z_list[1])

print(z_list[1][1])
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble



print(un_list1)

print(un_list2)

print(type(un_list2))
# LIST COMPREHENSION



# One of the most important topic of this kernel

# We use list comprehension for data analysis often

# list comprehension : collapse for loops building lists into a single line

# Ex : num1 = [1,2,3] and we want to make it num2 = [2,3,4]. This can be done with for loop. However it is unnecessarily long.

# We can make it one line code that is list comprehension



# Example



num1 = [1,2,3]

num2 = [i + 1 for i in num1] # list comprehension

print(num2)



# i+1 : list comprehension syntax

# for i in num1 for loop syntax

# i : iterartor

# num1 : iterable object

# Conditional on iterable



num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1] # else i-5 if i <7 means if i < 7 then i = i-5

print(num2)
# Lets return movie csv and make one more list comprehension example

# Lets classify movies whether they have high or low revenue. Our threshold is average revenue



threshold = sum(data.revenue)/len(data.revenue)

print(threshold)

data["revenue_level"] = ["high" if i > threshold else "low" for i in data.revenue]

data.loc[:10,["revenue_level","revenue"]]
# data frames from dictionary

title = ['Avatar', 'G.O.R.A']

genres = ['Action', 'Adventure']

list_label = ['title','genres']

list_col = [title,genres]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df['producer'] = ["James Cameron", "Cem Yılmaz"]

df['revenue'] = [760507625,26200000]

df
# Broadcasting

df['budget'] = 0 # Broadcasting entire column

df
# plotting all data

data1 = data.loc[:,["budget","revenue"]]

data1.plot()

plt.show()
# subplots

data1.plot(subplots = True)

plt.show()
# scatter

data1.plot(kind = "scatter", x = 'budget', y = 'revenue')

plt.show()
# hist plot

data.plot(kind = 'hist', y = 'popularity', bins = 50,range = (0,5), normed = True)

# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data.plot(kind = 'hist', y = 'popularity', bins = 50,range = (0,5), normed = True, ax = axes[0])

data.plot(kind = 'hist', y = 'popularity', bins = 50,range = (0,5), normed = True, ax = axes[1], cumulative = True)

plt.savefig('graph.png')

plt
time_list = ["2010-04-12","2008-04-25"]

print(type(time_list[1])) # as you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
data2 = data.head()

data2
data2 = data2.set_index("release_date")

data2
print(data2.loc["2009-12-10"])

print(data2.loc["2009-12-10":"2012-07-16"])
newdf = pd.to_datetime(data2.index)

data2.index = newdf

data2
data2.resample("A").mean()
# INDEXING DATA FRAMES

print(data["budget"][1]) # index using square bracjet

print(data.budget[1])    # using column attribute and row label

print(data.loc[1,["budget"]])   # using loc accessor
# selecting some columns

data[["budget","original_title"]].head()
# SLICING DATA FRAME

# difference between selecting columns: series and dataframes

print(type(data["budget"])) #series

print(type(data[["budget"]])) #dataframes
# Slicing and indexing series



data.loc[0:9,"budget":"original_title"]
# from something to end



data.loc[0:4,"original_title":]
# FILTERING DATA FRAMES

boolean = data.budget > 200000000

data[boolean]
# Combining filters



first_filter = data.budget > 200000000

second_filter = data.revenue > 1000000000

data[first_filter & second_filter]
# filtering column based others



data.budget[data.vote_average > 7.8]
# Plain python function

def div(n):

    return n/2

data.vote_average.apply(div).head()
# or we can use lambda function



data.vote_average.apply(lambda n: n/2).head()
# Defining column using other columns

data["profit"] = data.revenue - data.budget

data[["original_title","budget","revenue","profit"]].head()

# INDEX OBJECTS AND LABELED DATA



# our index name is this;

print(data.index.name)

#Lets change it

data.index.name = "index_name"

data.head()
data5 = data.copy()

# Lets make index start from 100

data5.index = range(100,4903,1)

data5.head()
# HIERARCHICAL INDEXING

# Setting index : production_companies is outer production_countries is inner index

data1 = data.set_index(["production_countries","production_companies"])

data1.head(100)
# PIVOTING DATA FRAMES

# pivoting : reshape tool



dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"], "response":[10,25,30,50],"age":[14,53,21,63]}

df = pd.DataFrame(dic)

df

# pivoting

df.pivot(index = "gender",columns="treatment",values="response")
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexis

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position



df2 = df1.swaplevel(0,1)

df2
# MELTING DATA FRAMES

# Reverse of pivoting



# df.pivot(index="treatment",columns="gender",values="response")



pd.melt(df,id_vars="treatment",value_vars = ["age","response"])



# variable and value default adding dataframe. We assign this values with age and response datas
# CATEGORICALS AND GROUPBY



df.groupby("treatment").mean()
df.groupby("gender").age.mean()
df.groupby("treatment")[["age","response"]].max()