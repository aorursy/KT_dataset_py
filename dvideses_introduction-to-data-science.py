# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization tools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/2015.csv")
data.columns
data.info()
data.describe()
data.head()
# The first 5 samples of the data set
data.head(10)
data.tail()
# The last 5 samples of the data set
data.dtypes
data.corr()
#Correlation
#Correlation Map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# Histogram
# bins = number of bar in figure
data.Generosity.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.xlabel("Generosity")
plt.title("HİSTOGRAM")
#Scatter Plot
data.plot(kind='scatter', x='Happiness Score', y='Economy (GDP per Capita)',alpha = 0.5,color = 'blue')
plt.xlabel('Happiness Score')              # label = name of label
plt.ylabel('Economy (GDP per Capita)')
plt.title('Scatter Plot')            # title = title of plot
# Comparison operator
print(5 > 3)
print(10!=9)
# Boolean operators
print(True and True)
print(False and True)
print(True or False)
print(False or True)

#Series
data=pd.read_csv("../input/2015.csv")
series=data["Happiness Rank"]
print(type(series))
#Data Frame
dataframe=data[["Happiness Rank"]]
print(type(dataframe))
x= data["Happiness Score"] > 7
data[x]
x= data["Happiness Score"] > 7
data[x]
data.head()
data[ (data["Happiness Score"]> 5) & (data["Freedom"]> 0.35) ]
# logical_and() function
data[np.logical_and(data["Health (Life Expectancy)"]> 0.94, data["Happiness Score"]>7 )]
# select columns by name
data.filter(items=['Country', 'Happiness Score'])
data.head(10)
a = 0
while a < 5:
    print("Coding is a learned skill that I believe anyone can acquire.",a)
    a += 1
# Stay in loop if condition( i is not equal 5) is true
lis = [15,25,35,45,55]
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
dictionary = {'Turkey':'İstanbul',' South Korea':'Seoul'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')
# For pandas we can achieve index and value
for index,value in data[['Country']][0:1].iterrows():
    print(index," : ",value)
# Countries bring all the lines of the column
print(data ["Country"])
# Countries bring all the lines of the column
data.loc[ :  , "Country"]
print(data.Region)
print(data.loc[ :3,"Happiness Rank"])
print(data.loc[ :3, ["Country", "Happiness Score"] ])
print(data.loc[ :3, "Country" : "Happiness Score" ])
# Example of list comprehension
num1 = [2,3,4]
num2 = [i + 1 for i in num1 ]
print(num2)
# lets return world happiness report csv and make one more list comprehension example
threshold = sum(data["Happiness Score"])/len(data["Happiness Score"])
data["score_level"] = ["high" if i > threshold else "low" for i in data["Happiness Score"]]
data.loc[:10,["score_level","Happiness Score"]] 
threshold = sum(data["Happiness Score"])/len(data["Happiness Score"])
print(threshold)
# Example of a user-defined function
def add_numbers(x,y):
    output = x + y
    return output

num1 = 5
num2 = 6

print("The sum is", add_numbers(num1, num2))
def tuble_ex():
    """ return defined t tuble"""
    t = (10,20,30)
    return t
a,b,c = tuble_ex()
print(a,b,c) 
x = 5
def f():
    x = 7
    return x
print(x)      # x = 5 global scope
print(f())    # x = 7 local scope
# What if there is no local scope
x = 5
def f():
    y = 10+x        # there is no local scope x
    return y
print(f())         # it uses global scope x
# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope
import builtins
dir(builtins)
def number(x):
    def add(y):
        return x + y
    return add

output = number(5)
print(output(12)) 
#Default Arguments
def x (a, b = 10):
    y = a*b 
    return y
print(x(5))
#Flexible Arguments
# *args
def f(*args):
    z = 1
    for i in args:
        z *= i
    print(z)

f(4, 5)
f(2, 3, 4)
# We used *args to send a variable-length argument list to our function, we were able to pass in as many arguments as we wished into the function calls.
def calculate(x):
    output=x*x
    return output
print(calculate(3))
calculate = lambda x: x * x
# Output: 10
print(calculate(3))
number_list = [5,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))
# iteration example
name = "Fenerbahçe"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration