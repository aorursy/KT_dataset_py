# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns #visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/pokemon.csv")
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f", ax = ax)
plt.show()
data.head(10)
data.columns
#Line Plot
#color = color, label = label, linewidth = width of the line, alpha = opacity, grid = grid, 
#linestyle = style of the line
data.Speed.plot(kind="line", color ='b', label ='Speed', linewidth =1, alpha =.5, 
                grid=True, linestyle=':' )
data.Defense.plot(kind='line', color='r', label='Defense', Linewidth =1, alpha=.5,
                 grid=True, linestyle='-.')
plt.legend(loc='upper right')   # legend = puts label into plot
plt.xlabel('x axis')           # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')                  # title = title of plot


plt.show()
#Scatter Plot
# x = attack, y = defense
data.plot(kind='scatter',x= 'Attack', y='Defense', alpha=.5,color='r', grid= True)
plt.title('Attack Defense Scatter Plot')         # title = title of plot
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.show()
#Histogram
#bin is number of bars in histogram
data.Speed.plot(kind='hist', color='y', bins= 50, figsize=(5,5))
plt.show()
#clf() is a method to clean the plot
data.Speed.plot(kind='hist', color='y', bins= 50, figsize=(5,5))
plt.clf()
# we dont see the plot cos of clf()
#create dictionary and look its keys and values
dictionary={'spain':'madrid', 'usa':'vegas'}
print(dictionary)
print(dictionary.keys())
print(dictionary.values())
# update existing entry
dictionary['spain'] = 'malaga'
print(dictionary)
# Add new entry
dictionary['france'] = 'paris'
print(dictionary)
# remove entry with key 'spain'
del dictionary['spain']
print(dictionary)
# check include or not
print('france' in dictionary)
# remove all entries in dict
dictionary.clear()
print(dictionary)

del dictionary # delete the dictionary from the memory
data = pd.read_csv('../input/pokemon.csv')
series = data['Defense']  
print(type(series))

dataFrame= data[['Defense']]
print(type(dataFrame))

#comparison
print(2>3)
#Boolean operators
print(True & False)
print(True or False)
# 1 - Filtering Pandas data frame
x=data['Defense']>200
data[x]

data[np.logical_and(data['Defense']>200, data['Attack']>100)]

#We can do it another way which is:
data[(data['Defense']>200) & (data['Attack']>100)]

# Stay in loop if condition( i is not equal 5) is true. 
i= 0
while i != 5: 
    print("i is :" ,i)
    i+=1
print (i, "is the last")
# Stay in loop if i is in the list
lis=[1,2,3,4,5]
for i in lis:
    print ("i is ", i)
print ("Thats it!")
# we use enumerate for lists
for index, value in enumerate(lis):
    print(index, ":" , value)    
#For dictionaries instead of enumerate, we use item
dictionary={'spain':'madrid', 'usa':'vegas'}
for key, values in dictionary.items():
    print(key, ':' , values)    
    
#for Pandas
for index, value in data[['Defense']][0:5].iterrows():
    print(index, ':' ,value )
data.head(10)
#tuble   tuble = (1,2,3)
#sequence of immutable python objects. Values cant be modified. 
# Function to unpack tuble into several variables like a,b,c 
def tuble_unpack():
    t=(1,2,3)
    return t
a,b,c = tuble_unpack()
print(a,b,c)
#It is same with 
t=(1,2,3)
a,b,c = t
print(a,b,c)
#we only wanted to show how functions defined

x=2 #Global variable (Global scope)
def f():
    x=3  # local variable(local scope)
    return x
print(x)
print(f())
#if there is no local variable, It will take the global one
def g():
    return x
print (g())

# How can we learn what are built in scope
import builtins
dir(builtins)
#LAMBDA FUNCTION
#Faster way of writing function

square = lambda x: x**2
print (square(2))

tot= lambda x,y,z : x+y+z
print(tot(2,3,4))
#ANONYMOUS FUNCTION-The difference is it can take more than one variable

alist = [1,2,3]
x = map(lambda x: x**2, alist) #gives the value for x one by one by usşng each elements of the alist
print(list(x))  #we need to show all the x values in a list 
#zip it doesnt work with strings but only integers. Why?
list1 = [1,2,3,4]
list2 = [5,6,7,8]
zip12 = zip(list1, list2)
print(zip12)
zip_list = list(zip12)
print(zip_list)

#unzip

unzip = zip(*zip_list)
unlist1, unlist2 = list(unzip) #unzip returns tuble. Converted to list. 
print(unlist1)
print(unlist2)
print (type(unlist1))

list1 = [1,2,3]
list2 = [2*i for i in list1]  #list of comprehension
print(list2)
#Let's see how we use list comprehension in data analysis
#Lets find the speeds and label them high or low according to their posion regarding to avarage speed
threshold = sum(data.Speed) / len(data.Speed)
print('Threshold is: ', threshold)
data['speed_level'] = ['high' if i >threshold else 'low' for i in data.Speed]
data.loc[:10,['speed_level','Speed']]

print(data['Type 1'].value_counts(dropna=False))
data.describe()
data.boxplot(column='Attack',by = 'Legendary')
plt.show()
newdata = data.head()
newdata
meltedone = pd.melt(frame=newdata, id_vars = 'Name', value_vars=['Attack', 'Defense'])
meltedone
#PIVOT

meltedone.pivot(index = 'Name', columns = 'variable', values='value')
#CONCATENATING 

data1 = data.head()
data2= data.tail()
concatenatedone = pd.concat([data1,data2], axis=0, ignore_index =True ) # axis = 0 : To add below
concatenatedone
# data1= data['Attack'].head()
data2 = data['Defense'].head()
cncatened_columns =pd.concat([data1,data2], axis=1)
cncatened_columns
#DATA TYPES:
#Five basic data types.

data.dtypes
data['Type 1'] = data['Type 1'].astype('category')
data.dtypes


