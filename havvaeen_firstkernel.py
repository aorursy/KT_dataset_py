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
data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
data.info()
data.corr()
#correlation map

f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()

data.head()
data.tail()
data.columns


data.year.plot(kind='line', color='g', label='year', linewidth=1, alpha=0.5, grid=True, linestyle=':')

data.population.plot(color='r', label='population', linewidth=1, alpha=0.5, grid=True, linestyle='-.')

plt.legend(loc='upper right') #puts label into plot

plt.xlabel('x axis') #label= name of label

plt.ylabel('y axis') 

plt.title('Line Plot') #title= title of plot

plt.show()

#scatter plot

#x = year

#y = population

data.plot(kind='scatter', x='year', y='population', alpha=0.5, color='red')

plt.xlabel('year')

plt.ylabel('population')

plt.title('Year Population No Scatter Plot')

plt.show()
#histogram

#bins=number of bar in figure

data.suicides_no.plot(kind='hist', bins=50, figsize=(12,12))

plt.show()
#clf() = cleans it up again you can start a fresh

data.suicides_no.plot(kind='hist', bins=50)

plt.clf

#BUT we cannot see plot due to clf()

#create dictionary and look its keys and values

dictionary ={'izmir': 'cesme', 'antalya':'belek'}

print(dictionary.keys())

print(dictionary.values())
#Keys have to be immutable objects like string, boolean, float, integer or tuples

#List is not immutable.

#Keys are unique.

dictionary['izmir']='dikili' #uptade existing entry

print(dictionary)

dictionary['mugla']='bodrum' #add new entry

print(dictionary)

del dictionary['izmir'] #remove entry with key 'izmir'

print(dictionary)

print('mugla' in dictionary) #check include or not

print('izmir' in dictionary)

dictionary.clear() #remove all entries in dict

print(dictionary)
#In order to run all code you need to take comment this line

#del dictionary #delete entire dictionary

print(dictionary) #it gives error because dictionary is deleted.
data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
series = data['age'] # data['suicide_no'] =series 

print(type(series))

data_frame = data [['age']] # data[['age']] = data frame

print(type(data_frame))

#comparison operator

print(3 > 2)

print(3!=2)

#Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas Data Frame

x = data ['year'] >2013 # There are 1840 suicide information who have higher year 2013 

data[x]

#2 - Filtering pandas with logical _ and

# There are 431 suicide information who have higher year than 2013 and higher population than 2000000

# Therefore we can also use'&' for filtering

data[np.logical_and(data['year']>2013, data['population']>2000000)]
# Stay in loop if condition (i is not equal 5) is true

i = 0

while i != 5 :

    print('i is: ',i)

    i+=1

print(i, 'is equal to 5')
# Stay in loop if condition (i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

#index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate (lis):

    print(index, " : ", value)

print('')



# For dictionaries

# We can use for loop to achieve key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'izmir':'urla', 'antalya':'kas'}

for key, value in dictionary.items():

    print(key," : ", value)

print('')



# For pandas we can achieve index and value

for index, value in data[['age']][0:1].iterrows(): #0 inclusive ; 1 exclusive

    print(index, " : ", value)
# Example of what we learn above 

def tuble_ex():

    """return defined t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# Guess print what

x = 2

def f():

    x = 3

    return x

print(x) # x = 2 global scope

print(f()) # x = 3 local scope
# How we learn what is built in scope

import builtins

dir(builtins)
# nested function

def square():

    """return square of value"""

    def add():

        """add two local variable"""

        x = 2

        y = 3

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

print(f(5, 4, 3))
# flexible arguments *args 

# args can be one or more

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1, 2, 3, 4)



# flexible arguments **kgwargs that is dictionary 

def f(**kwargs):

    """print key and value of dictionary"""

    for key, value in kwargs.items():

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)

        



square = lambda x : x**2

print(square(4))

tot = lambda x, y, z : x+y+z

print (tot(1,2,3))

number_list = [1,2,3]

y = map (lambda x : x**2, number_list)

print(list(y)) #sonucu liste olarak yazdır
# iteration example

name = "ronaldo"

it = iter(name)

print(next(it)) # print next iteration

print(*it) # print remaining iteration
# zip():zip lists

# zip example

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1, list2)

print(z)

z_list = list (z)

print(z_list)
un_zip = zip(*z_list)

un_list1, un_list2 = list(un_zip) # unzip returns tuple

print(un_list1)

print(un_list2)

print(type(un_list2))
num1 = [1,2,3]

num2 = [i + 1 for i in num1]

print(num2)
# Conditionals on iterable 

num1 = [5, 10, 15]

num2 = [i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]

print(num2)

data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

data.head()
data.tail()
data.columns
data.shape # Shape gives number of rows and columns in a tuple 
data.info()
print(data['country'].value_counts(dropna=False))
data.describe() #ignore null entries
data.boxplot(column = 'population', by = 'country')
#Firstly I create new data from suicides data to explain melt nore easily.

data_new = data.head()

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new, id_vars = 'suicides_no', value_vars = ['population','year'])

melted
# Index is country

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index ='suicides_no', columns = 'variable', values ='value')
# Firstly lets create 2 data frame

data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1, data2], axis = 0, ignore_index = True)

conc_data_row
data1 = data['population'].head()

data2 = data['country'].head()

conc_data_col = pd.concat([data1, data2], axis=1)

conc_data_col
data.dtypes
# Lets convert object to categorical and int to float

data['country'] = data['country'].astype('category')

data['suicides_no'] = data['suicides_no'].astype('float')
data.dtypes
data.info()
# Lets chech HDI for year 

data["HDI for year"].value_counts(dropna=False)

# As you can see, there are 19456 NAN value
# Lets drop nan values

data1 = data # Also we will use data to fill missing value, so I assign it to data1 variable.

data1["HDI for year"].dropna(inplace = True) # inplace=True means we do not assign it to new variable. Changes automatically assigned to data.
# Lets check with assert statement 

# Assert statement:

assert 1==1 #return nothing because it is true
# assert 1==2 #return to error
assert data['HDI for year'].notnull().all() # returns nothing because we do not have nan values
# Data frames from dictionaries

city = ["izmir", "antalya"]

population = ["11","12"]

list_label = ["city","population"]

list_col = [city, population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["region"] = ["ege","akdeniz"]

df
# Broadcasting

df["income"] = 0 # Broadcasting entire column

df
# Plotting all data

data1 = data.loc[:,["population","HDI for year", "gdp_per_capita ($)", "year"]]

data1.plot()

plt.show()

# it is confusing
# subplot

data1.plot(subplots = True)

plt.show()
# scatter plot

data1.plot(kind = "scatter", x = "population", y = "HDI for year")

plt.show()
# hist plot

data1.plot(kind = "hist", y="year", bins = 50, range = (0,250), normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows = 2, ncols = 1)

data1.plot(kind = "hist", y = "year", bins = 50, range = (0,250), normed = True, ax = axes[0])

data1.plot(kind = "hist", y = "year", bins = 50, range = (0,250), normed = True, ax = axes[1], cumulative = True)

plt.savefig('graph.png')

plt

data.describe()
time_list = ["1992-03-08", "1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# However we want to it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning 

import warnings

warnings.filterwarnings("ignore")

# in order to practice lets take head of suicide data and add it a time list

data2 = data.head()

date_list = ["2019-08-23", "2019-11-15", "2019-12-29", "2020-01-11", "2020-02-11"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2 = data2.set_index("date")

data2
# now we can select according to our date index

print(data2.loc["2020-02-11"])

print(data2.loc["2019-12-29":"2020-02-11"])
# we will use data2 that we create at previous part

data2.resample("A").mean()
# lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# in real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# we can interpolate with mean().

data2.resample("M").mean().interpolate("linear") # mean() yerine first() : interpolate from first value
# read data

data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

#data= data.set_index("#") #NEDEN HATA

data.head()
# indexing using square brackets

data["population"][1]
# using column attribute and row label

data.population[1]
# using loc accessor

data.loc[1, ["population"]]
# selecting only some columns

data[["population", "country"]]
# Difference between selecting columns: series and dataframes

print(type(data["population"])) #series

print(type(data[["population"]])) # data frames
# slicing and indexing series

data.loc[1:10, "year":"population"]  # 10 and populatşon inclusive
# reverse slicing

data.loc[10:1:-1, "year":"population"]
# from something to end

data.loc[1:10, "population":]
# Creating boolean series

boolean = data.population > 30000

data[boolean]
# Combining filters

first_filter = data.population > 30000

second_filter = data.year > 2015

data[first_filter & second_filter]
# filtering column based others

data.population[data.year > 2015]
# plain python functions

def div(n):

    return n/2

data.population.apply(div)
# or we can use lambda function

data.population.apply(lambda n : n/2)
# defining column using other columns

data["total_power"] = data.population + data.year

data.head()
# our index name is this:

print(data.index.name)

#lets change it

data.index.name = "index_name"

data.head()
# overwrite index

# if we want to modify index we need to change all of them.

data.head()

#first copy of our data to data3 then change index 

data3 = data.copy()

#lets make index start from 100. it is not remarkable change but it is just example

data3.index = range(100,27920,1)

data3.head()
data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

data.head()
# setting index: country is outer year is inner index

data1 = data.set_index(["country", "year"])

data1.head(100)
dic = {"treatment":["A","A","B","B"], "gender":["F","M","F","M"], "response":[10,45,5,9],"age":[12,45,78,96]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index ="treatment",columns = "gender", values = "response")
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
# df.pivot(index="treatment",columns="gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# We will use df

df
# according to treatment take means of other features

df.groupby("treatment").mean() # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max() 
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 
df.info()

# as you can see gender is object

# however if we use groupby, we can convert it categorical data.

##df["gender"] = df["gender"].astype("category")

#df["treatment"] = df["treatment"].astype("category")

#df.info()