# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data
data.info()
data.corr()
b,ax = plt.subplots(figsize=(25,25))

sns.heatmap(data.corr(), annot=True, linewidths=5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
#Line Plot

#  color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Reactions.plot(kind = 'line', color = 'g', label = 'Reactions', linewidth = 1, alpha = 0.5,grid = True,linestyle = ':')

data.Overall.plot(color = 'r',label = 'Overall', linewidth = 1, alpha = 0.5, grid = True, linestyle = '--')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
#Scatter Plot

#x = Dribbling , y = Strength

data.plot(kind='scatter' , x='Overall', y='Potential', alpha = 0.3,color = 'orange')

plt.xlabel('Overall')

plt.ylabel('Potential')

plt.title('Potential&Overall Scatter Plot')

#Histogram 

# bins = number of bar in figure

data.Potential.plot(kind='hist',bins = 50, figsize=(12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Potential.plot(kind = 'hist',bins = 50)

plt.clf()

#creating dictionary nd looking its keys and values

dictionary = {'turkey' : 'ankara', 'france' : 'paris'}

print(dictionary.keys())

print(dictionary.values())
dictionary['turkey'] = "istanbul" # updating entry

print(dictionary)

dictionary['england'] = 'london' # adding new entry

print(dictionary)

#del dictionary['france'] removing entry with keyword = france

#print('france' in dictionary) checking include or not

#dictionary.clear() removing all entries

data = pd.read_csv('../input/data.csv')
series = data['Dribbling']

print(type(series))

data_frame = data[['Dribbling']]

print(type(data_frame))
x = data['Acceleration']>95

data[x]
data[np.logical_and(data['Acceleration']>94, data['Dribbling']>85 )]
data[np.logical_or(data['Acceleration']>94, data['Dribbling']>85 )]
data[(data['Acceleration']>94) & (data['Dribbling']>85)]

i = 5

while i != 10: 

    print('i is: ',i)

    i+=1

print(i,' is equal to 10')
lis = [1,2,4,5,6,7]

for i in lis:

    print('i is: ',i)

print('')



dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['BallControl']][0:1].iterrows():

    print(index," : ",value)

def tuble_ex():

    """ return defined t tuble"""

    t = (1,2,3,4,5)

    return t

a,b,c,d,e = tuble_ex()

print(a,b,c,d,e)
x = 2

def f():

     x = 3

     x = x * 5

     return x

print(x)      # x = 2 global scope

print(f())   # x = 15 local scope
# What if there is no local scope

x = 9

def f():

    a = x * 7        # there is no local scope x

    return a

print(f())         # it uses global scope x

# How can we learn what is built in scope

import builtins

dir(builtins)
#nested function

def square():

    def add():

        

        a = 7

        b = 8

        c = a + b

        return c

    return add()**b

print(square())    
# default arguments

def f(a, b = 4, c = 2):

    y = a + b + c

    return y

print(f(10))

# what if we want to change default arguments

print(f(5,4,3))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(99)

print("")

f(4,3,1,4)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    for key, value in kwargs.items():

        print(key," ", value)

f(country = 'France', capital = 'Paris', population = 123456)
# lambda function

square = lambda x: x**5     # where x is name of argument

print(square(3))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
numberList = [1,2,3,4,5]

y = map(lambda x:x**6,numberList)

print(list(y))
# iteration example

name = "oktay"

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

num1 = [1,5,10]

num2 = [i + 3 for i in num1 ]

print(num2)

# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**3 if i == 20 else i-7 if i < 9 else i+5 for i in num1]

print(num2)
# lets return fifa 19 complete football players csv and make one more list comprehension example

# lets classify football players whether they have high or low vision. Our players is average vision.

Player_Vision = data.Vision.sum()/len(data.Vision)

data["Vision_level"] = ["high" if i > Player_Vision else "low" for i in data.Vision]

data.loc[:10,["Vision_level","Vision"]] 
data = pd.read_csv('../input/data.csv')

data.head()  # head shows first 5 rows
# it shows last 5 rows

data.tail()
#which features we have own 

data.columns

# number of rows & columns 

data.shape

# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
print(data['Nationality'].value_counts(dropna = False))
data.describe() # ignoring null entries
data.boxplot(column='Overall',by = 'Potential',figsize=(15,15))
# Firstly I create new data from footballers data to explain melt nore easily.

data_new = data.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Age','Overall','Potential'])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Name', columns = 'variable',values='value')
# Firstly lets create 2 data frame

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) 

conc_data_row
data1 = data['Overall'].head()

data2= data['Potential'].head()

data3= data['Age'].head()

conc_data_col = pd.concat([data1,data2,data3],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
# converting object(str) to categorical and int to float

data['Nationality'] = data['Nationality'].astype('category')

data['Age'] = data['Age'].astype('float')
#We need to focus that Age & Nationality 

data.dtypes
data.info()
data["Skill Moves"].value_counts(dropna = False)
#Dropping NaN Values

data1 = data

data1["Skill Moves"].dropna(inplace = True)

# True means we do not assign it to new variable. Changes automatically assigned to data

# We can control with assert statement
assert 1 == 1 # Simple Example

#assert 1 == 2 
assert data['Skill Moves'].notnull().all() # It return nothing because nan values are dropped.
data["Skill Moves"].fillna('empty',inplace = True)
assert data['Skill Moves'].notnull().all()
# Plotting all data 

data1 = data.loc[:,["Age","Overall","Potential"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Overall",y = "Potential")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Overall",bins = 50,range= (40,100),normed = True)
# hist plot  

data1.plot(kind = "hist",y = "Potential",bins = 50,range= (40,100),normed = True)

# 3 sigma rule valid in these graphs.
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Overall",bins = 50,range= (40,100),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Potential",bins = 50,range= (40,100),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
print(type(data["Overall"])) # series

print(type(data[["Overall"]])) # data frames

#Slicing and indexing series

data.loc[4030:4040, "Overall":"Potential"]
data.loc[4040:4030:-1,"Overall":"Potential"] 
#From something to end

data.loc[4030:4040,"Marking":]
# Creating boolean series

boolean = data.Age >27

data[boolean]
#Combining Filters

first_filter = data.Age > 30

second_filter = data.Strength > 90

data[first_filter & second_filter]
# Filtering column based others

data.Overall[data.Age < 17]
# plain python functions

def div(n):

    return n/4

data.Strength.apply(div)
#Lambda function

data.Strength.apply(lambda n : n/4)
#Defining column using other columns

data["Potential_Overall"] = data.Potential / data.Overall

data.head()
#our index name is this

print(data.index.name)



data.index.name = "index_name"

data.head()
data.head()
# Setting index : overall is outer potential is inner index

data1 = data.set_index(["Overall","Potential"])

data1.head(100)
dic = {"treatment":["A","A","C","C"],"gender":["F","M","M","F"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
#pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1

df1.unstack(level=0)

# level = index 
df1.unstack(level=1)
# changing inner & outer level index position

df2 = df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
df.groupby("treatment").mean() # aggreation / reduction method

# also sum, std, max, or min we can use it.
df.groupby("treatment").age.max()
df.groupby("treatment")[["age","response"]].min()
df.info()