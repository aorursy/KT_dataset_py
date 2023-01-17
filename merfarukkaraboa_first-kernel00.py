# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/fifa19/data.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Creating Data Frame

data = pd.read_csv('../input/fifa19/data.csv')
#Getting Information

data.info()
#Data correlation List

data.corr()
#correlation map

f,ax = plt.subplots(figsize=(30, 30))

sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.2f',ax=ax)

plt.show()
#Geeting First 20 Object 

data.head(20)
data.columns
#Age younger than 18 and overall bigger than 70 players

data[(data['Age']<18) & (data['Overall']>70)]
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Potential.plot(kind = 'line', color = 'g',label = 'Potential',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Overall.plot(color = 'r',label = 'Overall',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = BallControl, y = ShortPassing

data.plot(kind='scatter', x='BallControl', y='ShortPassing',alpha = 0.5,color = 'red')

plt.xlabel('BallControl')              # label = name of label

plt.ylabel('ShortPassing')

plt.title('BallControl-ShortPassing Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.Age.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Age.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'Hans' : '18','Refik' : '32'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Hans'] = "19"    # update existing entry

print(dictionary)

dictionary['Miranda'] = "25"       # Add new entry

print(dictionary)

del dictionary['Refik']              # remove entry with key 'Refik'

print(dictionary)

print('Miranda' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
data = pd.read_csv('../input/fifa19/data.csv')
series = data['Potential']        # data['Potential'] = series

print(type(series))

data_frame = data[['Potential']]  # data[['Potential']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['Agility']>80     # There are 18142 players who have higher Agility value than 80

data[x]
# 2 - Filtering pandas with logical_and

# There are only 1 player who have higher Balance value than 75 and higher LongPassing value than 90

data[np.logical_and(data['Balance']>75, data['LongPassing']>90 )]
# Stay in loop if condition( i is not equal 9) is true

i = 0

while i != 9:

    print('i is: ',i)

    i +=1 

print(i,' is equal to 9')
# Stay in loop if condition( i is not equal 9) is true

lis = [1,2,3,4,5,6,7,8,9]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'Hans':'19','Miranda':'25'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['LongPassing']][0:1].iterrows():

    print(index," : ",value)



# example of what we learn above

def tuble_ex():

    """ return defined t tuble"""

    t = (5,9,7)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# guess print what

x = 0

def f():

    x = 1

    return x

print(x)      # x = 0 global scope

print(f())    # x = 1 local scope
# What if there is no local scope

x = 1

def f():

    y = 9*x        # there is no local scope x

    return y

print(f())         # it uses global scope x

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope

import builtins

dir(builtins)
#nested function

def square():

    """ return square of value """

    def add():

        """ add four local variable """

        a = 9

        x = 2

        y = 3

        z = 1

        q = x * z  + a / y

        return q

    return add()**2

print(square())    
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))

# what if we want to change default arguments

print(f(11,22,33))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(9,7,5,3)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(Hans='19', Miranda='25' ,Refik='32')
# lambda function

square = lambda x: x**2     # where x is name of argument

print(square(9))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(7,9,11))
number_list = [1,2,3]

y = map(lambda x:x**3,number_list)

print(list(y))
# iteration example

name = "messi"

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

num1 = [9,19,29]

num2 = [i + 1 for i in num1 ]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**3 if i == 5 else i-5 if i < 12 else i+5 for i in num1]

print(num2)
# lets return fifa 19 csv and make one more list comprehension example

# lets classify players whether they have high or low overall. Our threshold is average overall.

threshold = data.Overall.mean()

data["Overall_level"] = ["high" if i > threshold else "low" for i in data.Overall]

data.loc[:10,["Overall_level","Overall"]] # we will learn loc more detailed later
data = pd.read_csv('../input/fifa19/data.csv')

data.head()  # head shows first 5 rows
# tail shows last 5 rows

data.tail()
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
# For example lets look frequency of player types

print(data.Position.value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 112 santrafor players or 70 Goolkeeper players
# For example max age is 45 or min potential is 48

data.describe() #ignore null entries
# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='Potential',by = 'Age')
# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Overall','Potential'])

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
data1 = data['Overall'].head()

data2= data['Potential'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
# lets convert object(str) to categorical and int to float.

data['Name'] = data['Name'].astype('category')

data['Overall'] = data['Overall'].astype('int64')
data.dtypes
# Lets look at does Fifa19  data have nan value

# As you can see there are 18207 entries. However 'Loaned From' has 1264 non-null object.

data.info()
# Lets chech 'Loaned From'

data["Loaned From"].value_counts(dropna =False)

# As you can see, there are 16943 NAN value
# Lets drop nan values

data1=data   # also we will use data to fill missing value so I assign it to data1 variable

data1["Loaned From"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert  data['Loaned From'].notnull().all() # returns nothing because we drop nan values
data["Loaned From"].fillna('empty',inplace = True)
assert  data['Loaned From'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

# assert data.columns[1] == 'ID'

# assert data.Age.dtypes == np.float
# data frames from dictionary

country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["capital"] = ["madrid","paris"]

df
# Broadcasting

df["income"] = 0 #Broadcasting entire column

df
# Plotting all data 

data1 = data.loc[:,["Age","Agility","Potential"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Age",y = "Potential")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Potential",bins = 50,range= (0,250),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Potential",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Potential",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

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

# In order to practice lets take head of Fifa19 data and add it a time list

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
# We will use data2 that we create at previous part

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
## read data

data = pd.read_csv('../input/fifa19/data.csv')

data= data.set_index("Unnamed: 0")

data.head()
# indexing using square brackets

data["Age"][1]
# using column attribute and row label

data.Age[1]
# using loc accessor

data.loc[1,["Age"]]
# Selecting only some columns

data[["Age","Potential"]]
# Difference between selecting columns: series and dataframes

print(type(data["Overall"]))     # series

print(type(data[["Overall"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"Overall":"Marking"]   # 10 and "Marking" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"Overall":"Marking"] 
# From something to end

data.loc[1:10,"Potential":] 
boolean = data.Age < 27

data[boolean]
# Combining filters

first_filter = data.Age > 25

second_filter = data.Potential >75

data[first_filter & second_filter]
# Filtering column based others

data.Age[data.Overall<47]
# Plain python functions

def div(n):

    return n/2

data.Potential.apply(div)
# Or we can use lambda function

data.Potential.apply(lambda n : n/2)
# Defining column using other columns

data["max improve point"] = data.Potential - data.Overall

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("ID")

# also you can use 

# data.index = data["#"]
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/fifa19/data.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : Age is outer Potential is inner index

data1 = data.set_index(["Age","Potential"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # how to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
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
