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

        from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/world-happiness/2015.csv')
data.info()
data.corr()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Freedom.plot(kind='line',color='r',label='FREEDOM',linewidth=1,alpha=0.8,grid= True,linestyle=':')

data.Family.plot(color = 'r',label = 'Family',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.show()
# Scatter Plot 

# x = Freedom, y = Family

data.plot(kind='scatter', x='Freedom', y='Family',alpha = 0.5,color = 'blue')

plt.xlabel('Freedom')              # label = name of label

plt.ylabel('Family')

plt.title('Freedom Family Scatter Plot')            # title = title of plot

plt.show()

# Histogram

# bins = number of bar in figure

data.Family.plot(kind = 'hist',bins = 50,figsize = (18,18))

plt.show()
#create dictionary and look its keys and values

dictionary = {'Turkey' : 'Peace','Syria' : 'Terrorism'}

print(dictionary.keys())

print(dictionary.values())

# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary['Turkey']='tranquility'

print(dictionary)

dictionary['persia']='nuclear'

print(dictionary)

del dictionary['Syria']              # remove entry with key 'spain'

print(dictionary)

print('france' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)
series = data['Family']        # data['Defense'] = series

print(type(series))

data_frame = data[['Family']]  # data[['Defense']] = data frame

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data['Happiness Score']>7.406    # There are only 5 country that have higher happiness value than 7.406

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['Happiness Score']>7.406, data['Freedom']>0.63 )]
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
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

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')





# example of what we learn above

def tuble_ex():

    """ return defined t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# guess print what

x = 6

def f():

    x = 8

    return x

print(x)      # x = 6 global scope

print(f())    # x = 8 local scope
# What if there is no local scope

x = 4

def f():

    x=12

    y = 2*x        # there is local scope x

    return (y)

print(f())         # it uses global scope x

print(x)



# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
import builtins

dir(builtins)
#nested function

def square():

    """ return square of value """

    def add():

        """ add two local variable """

        x = 2

        y = 5

        z = x + y

        return z

    return add()**3

print(square())

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

num1 = [5,3,9]

num2 = [i + 5 for i in num1 ]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**3 if i == 10 else i-5 if i < 6 else i+5 for i in num1]

print(num2)
# lets return Happiness Range csv and make one more list comprehension example

# lets classify countries whether they have high or low happiness. Our threshold is average happiness.

threshold = sum(data.Freedom)/len(data.Freedom)

data["Freedom"] = ["high" if i > threshold else "low" for i in data.Freedom]

data.loc[:10,["Freedom","score"]] # we will learn loc more detailed later
data = pd.read_csv('../input/world-happiness/2015.csv')

data.head()  # head shows first 5 rows
data.tail()



data.columns
data.shape
data.info()
data.boxplot(column='Family',by = 'Freedom')
data_fon=data.head()

data_fon
melted = pd.melt(frame=data_fon,id_vars = 'Region', value_vars= ['Family','Freedom'])

melted
melted.pivot(index = 'Region', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['Family'].head()

data2= data['Freedom'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
# lets convert object(str) to categorical and int to float.

data['Happiness Rank'] = data['Happiness Rank'].astype('category')

data['Generosity '] = data['Generosity'].astype('float')
data.dtypes
# Lets look at does happiness data have nan value

# As you can see there are 158 entries. However this sample has 158 non-null object so it has not a null value

data.info()
# data frames from dictionary

country = ["Syria","Morocco"]

population = ["18","28"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
df["capital"] = ["sam","rabat"]

df
df["income"] = 0 #Broadcasting entire column

df
data.head()
data1 = data.loc[:,["Freedom","Family","Happiness Score"]]

data1.plot()

plt.show()
data1.plot(subplots=True)

plt.show()
data1.plot(kind = "scatter",x="Family",y = "Freedom")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Family",bins = 50,range= (0,250),normed = True)

plt.show()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Family",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Family",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt

plt.show()
data.describe()
time_list = ["1995-03-08","1996-04-10"]

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
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
# MANIPULATING DATA FRAMES WITH PANDAS
data["Freedom"][1]
print(type(data["Freedom"]))     # series

print(type(data[["Family"]]))   # data frames
data.loc[1:10,"Family":"Freedom"]   # 10 and "Defense" are inclusive
### TRANSFORMING DATA



# Plain python functions

def div(n):

    return n/2

data.Freedom.apply(div)
### INDEX OBJECTS AND LABELED DATA



# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()