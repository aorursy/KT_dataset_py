# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

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
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')

data.info()
# correlation map 

# correlation: featurelar arasındaki ilişkiyi anlamamızı sağlayan belli başlı parametrelerimiz vardır.Pozitif(evin oda sayısının artması ile evin fiyatının artması-) vs negatif(evin şehir merkezinden uzaklaştıkça fiyatının azalması) correlate

# Pozitif = 0<x<1, Negatif=-1<x<0, İlişki yok =0

data.corr()  # datanın correlation'ınının excel hali
f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind='line', color='g', label='Speed', linewidth=1, alpha=0.5, grid=True, linestyle=':')

data.Defense.plot(kind='line', color='r', label='Defense', linewidth=1, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc='upper right')  # legend = puts label into plot

plt.xlabel('x axis')  # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')  # title = title of plot

plt.show()
# Scatter Plot

# x=attack, y=defense

data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='red')

plt.xlabel('Attack')  # label= name of label

plt.ylabel('Defense')

plt.title('Attack Defense Scatter Plot')

plt.show()
# Histogram

# bins = number of bar in figure

data.Speed.plot(kind='hist', bins=50, figsize=(12, 12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Speed.plot(kind='hist', bins=50)

plt.clf()

# We cannot see plot due to clf()
dictionary = {'spain': 'madrid', 'usa': 'vegas'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['spain'] = "barcelona"  # update existing entry

print(dictionary)

dictionary['france'] = "paris"  # add new entry

print(dictionary)

del dictionary['spain']  # remove entry with key 'spain'

print(dictionary)

print('france' in dictionary)  # check include or not

dictionary.clear()  # remove all entries in dict

print(dictionary)

# In order to run all code you need to take comment this line

#del dictionary # delete entire dictionary

print(dictionary)  # will give  error because dictionary is deleted
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
series = data['Defense']  # series

print(series)

print('------------')

data_frame = data[['Defense']]  # data frame

print(data_frame)
# Comparison operator

print(3 > 2)

print(3 != 2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['Defense'] > 200, data['Attack'] > 100)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['Defense'] > 200) & (data['Attack'] > 100)]
# Stay in loop if condition ( i is not equal 5) is true

i = 0

while i != 5:

    print('i is: ', i)

    i += 1

print('i is equal to 5.')
# Stay in loop if condition( i is not equal 5) is true

liste = [1, 2, 3, 4, 5]

for i in liste:

    print('i is: ', i)

print('')



# Enumerate index and value of liste

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(liste):

    print(index, " : ", value)

print('')



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'spain': 'madrid', 'france': 'paris'}

for key, value in dictionary.items():

    print(key, " : ", value)

print('')



# For pandas we can achieve index and value

for index, value in data[['Attack']][0:10].iterrows():

    print(index, " : ", value)
# example of what we learn above

def tuble_ex():

    """ return defined t tuble"""

    t = (1, 2, 3)

    return t





a, b, c = tuble_ex()

print(a, b, c)

# Global vs Local Scope

x = 2





def f():

    x = 3

    return x





print(x)  # x=2 global scope

print(f())  # x=3 local scope

# What if there is no local scope

x = 5





def f():

    y = 2 * x  # there is no local scope x

    return y





print(f())  # it uses gloval scope x

# First local scope searched, then gloval scope searched, if two of them cannot b found lastly built in scope searched

# How can we learn what is built in scope

import builtins



dir(builtins)

# nested function

def square():

    def add():

        x = 2

        y = 3

        z = x + y

        return z



    return add() ** 2





print(square())

# default arguments

def f(a, b=1, c=2):

    y = a + b + c

    return y





print(f(5))

# what if we want to change default arguments

print(f(5, 4, 3))

# flexible arguments *args

def f(*args):

    for i in args:

        print(i)





f(1)

print('')

f(1, 2, 3, 4)





# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    for key, value in kwargs.items():

        print(key, ' : ', value)





f(country='spain', capital='madrid', population=123456)

# user defined function (long way)

def square(x):

    return x ** 2





print(square(5))

# lambda function(short way)

square = lambda x: x ** 2  # where x is none of argument

print(square(4))

total = lambda x, y, z: x + y + z  # where x,y,z are names of arguments

print(total(1, 2, 3))

number_list = [1, 2, 3]

y = map(lambda x: x ** 2, number_list)

print(list(y))

# iteration example

name = "ronaldo"

it = iter(name)

print(next(it))  # print next iteration

print(*it)  # print remaining iteration

list1 = [1, 2, 3, 4]

list2 = [5, 6, 7, 8]

z = zip(list1, list2)

print(z)

zipList = list(z)

print(zipList)

unzip = zip(*zipList)  # unzip the list

unzipList1, unzipList2 = list(unzip)  # unzip returns table

print(unzipList1)

print(unzipList2)

print(type(unzipList2))

# tuple to list

print(type(list(unzipList2)))



# example of list comprehension

num1 = [1, 2, 3]

num2 = [i + 1 for i in num1]

print(num2)

# Conditionals on iterable

num1 = [5, 10, 15]

num2 = [i ** 2 if i == 10 else i - 5 if i < 7 else i + 5 for i in num1]

print(num2)

# lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.Speed) / len(data.Speed)

data["speed_level"] = ['high' if i > threshold else 'low' for i in data.Speed]

data.loc[:10, ["speed_level", "Speed"]]

data.head()
# For example lets look frequenct of pokemon types

rint(data['Type 1'].value_counts(dropna=False))  # if there are nonen values that also be counted

# As it can be seen below there are 112 water pokemon or 98 normal pokemon

data.describe() # ignore null entries
# For example: compare attack of pokemons that are legendary or not

# Black line at top is max

# Blue line at top is 75%

# Green line is median(50%)

# Bluea line at bottom is 25%

# Black line at bottom is min

data.boxplot(column='Attack', by='Legendary')

# Firstly I create a new data from pokemons data to explain melt more easily.

dataNew = data.head()

print(dataNew)

# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=dataNew, id_vars='Name', value_vars=['Attack', 'Defense', 'Type 1'])

melted

# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index='Name', columns='variable', values='value')

# Firstly lets create 2 data frame

# ignore_index=True : we created new index

# Vertical Concat

data1 = data.head()

data2 = data.tail()

concateDataColumn = pd.concat([data1, data2], axis=0, ignore_index=True)  # axis =0 : adds dataframes in row

concateDataColumn

# Horizontal concat

data1 = data['Attack'].head()

data2 = data['Defense'].head()

concatDataColumn = pd.concat([data1, data2], axis=1)  # axis= 0 : adds dataframes in column

concatDataColumn

data.dtypes
# lets convert object(str) to categorical and int to float

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')

data.dtypes

# Lets look at does pokemon data have nan value

# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.

data.info()

# lets check type 2

data['Type 2'].value_counts(dropna=False)  # with dropna=False; we showed also NaN values in info

# there are 386 NaN value

# Lets drop NaN values

data1 = data

data1['Type 2'].dropna(inplace=True)  # we inplace=True we updated data after drop 'Type 2' column

# lets check with assert statement

# assert statement:

assert 1 == 1  # return nothing because it is true

# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert data['Type 2'].notnull().all()  # returns nothing we drop NaN values

data['Type 2'].fillna('empty', inplace=True)
assert data['Type 2'].notnull().all  # returns nothing because we do not have NaN values

# With assert statement we can check a lot of thing. For example

# assert data.columns[1] == 'Name'

# assert data.Speed.dtypes == np.int

# data frames from dictionary

country = ["Spain", "France"]

population = ["11", "12"]

list_label = ["country", "population"]

list_col = [country, population]

zipped = list(zip(list_label, list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df

# Add new columns

df['capital'] = ['madrid', 'paris']

df

# Broadcasting

df['income'] = 0  # Broadcasting entire column

df

# Plotting all data

data1 = data.loc[:, ["Attack", "Defense", "Speed"]]

data1.plot()

plt.show()

# subplots

data1.plot(subplots=True)

plt.show()

# scatter plot

data1.plot(kind="scatter", x="Attack", y="Defense")

plt.show()

# hist plot

# range : range in y axis

data1.plot(kind="hist", y="Defense", bins=50, range=(0, 250), normed=True)

plt.show()

# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2, ncols=1)

data1.plot(kind="hist", y="Defense", bins=50, range=(0, 250), normed=True, ax=axes[0])

data1.plot(kind="hist", y="Defense", bins=50, range=(0, 250), normed=True, ax=axes[1], cumulative=True)

plt.savefig('graph.png')

plt.show()

data.describe()
time_list =["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))
# In order to practice lets take head of pokemon data and add it a time list

data2=data.head()

date_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object=pd.to_datetime(date_list)

data2["date"]= datetime_object

# lets make date as index

data2=data2.set_index("date")

data2
# Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
# we will use data2 that we create at previous part

data2.resample("A").mean()

# data2.resample("A").sum()
# lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real, not created from us like data2) we can solve this problem with interpolate

# We can interpolate from first value

data2.resample("M").first().interpolate("linear")
# or we can interpolate with mean ()

data2.resample("M").mean().interpolate("linear")
# change data's index to '#' column

data = data.set_index("#")

data.head()
# indexing using square brackets

data["HP"][1]
# using loc accessor

data.loc[1,["HP"]]
# selecting only some columns

data[["HP","Attack"]]
# Difference between selecting columns: series and dataframes

print(type(data["HP"]))     # series

print(type(data[["HP"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"]
# From something to end

data.loc[1:10,"Speed":]
# Creating boolean series

boolean = data.HP > 200

data[boolean]
# Combining filters

first_filter = data.HP > 150

second_filter = data.Speed < 35

data[first_filter & second_filter]
# Filtering column based others

data.HP[data.Speed<15]
# Plain python functions

def div(n):

    return n/2



data.HP.apply(div)
# Or we can use lambda function

data.HP.apply(lambda n : n/2)
# Defining columns using other columns

data["total_power"] = data.Attack + data.Defense

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# Overrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data2 then change index

data2 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data2.index = range(100,900,1)

data2.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("#")

# also you can use 

# data.index = data["#"]
# lets read data frame one more time to start from beginning

data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : Type 1 is outer, Type 2 is inner index

data1 = data.set_index(["Type 1", "Type 2"])

data1.head(100)

# data1.loc["Fire", "Flying"] # how to use indexes
dic = { "treatment":["A","A","B","B"], "gender":["F","M","F","M"], "response": [10, 45, 5, 9], "age": [15, 4, 72, 65] }

df = pd.DataFrame(dic)

df

# pivoting

df.pivot(index="treatment",columns="gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexes

df1.unstack(level=0)

# we deleted treatment index
df1.unstack(level=0)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# We will use df

df

# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max()
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min()
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["gender"] = df["gender"].astype("category")

#df["treatment"] = df["treatment"].astype("category")

#df.info()