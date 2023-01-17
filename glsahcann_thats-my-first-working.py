# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv ('../input/countries of the world.csv')
print(data.columns)  #column names (feature)
data.info()  #gives information about the database
data.describe()  #gives information about numeric values
data.head()  #show the top five entries in the database
data.tail()  #gives the last five entries in the database
#Correlation
data.corr()  
#Correlation Map
plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5)
plt.show()
print(data.Country) 
#print(data.loc[:,"Country"])
#print(data.loc[:,:"Country"])
#print(data.iloc[:,0])
print(data.loc[:5,"Country"])
print(data.loc[:3,"Country":"Population"])
print(data.loc[:4,["Country","Population"]])
print(data.loc[::-1,:])
print(data.Region)
print(data.Region.unique())
#Changing Feature
new_columns_name = ['Country', 'Region', 'Population', 'Area', 'Pop_Density','Coastline', 'Net_migration', 'Infant_mortality', 'GDP','Literacy_(%)', 'Phones',
'Arable_(%)', 'Crops_(%)',  'Other_(%)', 'Climate', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry','Service']
data.columns = new_columns_name
print(data.columns)
mean_population = data.Population.mean()   #Calculate the average of column Population
data["Population_level"] = [ "low" if mean_population > each else "high" for each in data.Population ]  #Creates new column with population levels
print(data.loc[:5,["Country","Population","Population_level"]])
data.drop(["Service"],axis=1,inplace = True)  #delete column Service
print(data.columns)
#Filtering
data[(data['Population'] > 70000000)]
#Filtering
data[(data['Population'] > 70000000) & (data['Area'] == 780580)]
#Line Plot
data.Area.plot(kind = 'line', color = 'blue', label = 'Area' , linewidth = '1' , grid = True)
data.Population.plot(kind = 'line', color = 'red', label = 'Population' , linewidth = '1', grid = True , alpha = 0.8)
plt.legend()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()

#Line Plot
plt.plot(data.Area, color = 'blue', label = 'Area')
plt.plot(data.Population, color = 'red' , label = 'Population', alpha = 0.7, linestyle = ':')
plt.legend()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show

#Subplot
plt.subplot(2,1,1)
plt.plot(data.Area, color="blue", label="Area")
plt.ylabel("Area")
plt.legend()
plt.subplot(2,1,2)
plt.plot(data.Population, color="red", label="Population")
plt.ylabel("Population")
plt.legend()
plt.show()

#Scatter Plot
data.plot(kind='scatter', x='Area', y='Population',color = 'pink')
plt.xlabel('Area')              
plt.ylabel('Population')
plt.title('Scatter Plot')

#Scatter Plot
plt.scatter(data.Population, data.Area, color="pink")
plt.xlabel('Area')              
plt.ylabel('Population')
plt.title('Scatter Plot')

#Histogram
data.GDP.plot(kind = 'hist',bins = 50,figsize = (5,5))
plt.title("Histogram")
plt.show()
#Defined Function
def operation():
    """return defined x"""  #docstring: explanation for function
    x = 1
    return x
print(operation())

def operation_1(x):
    """return x variable 's square"""
    y = x**2
    return y
print(operation_1(2))
#Lambda Function
operation_1 = lambda x: x**2     #function is single line
print(operation_1(2))
#Anonymous Function -> uses lambda function
value_list = [0,1,2]
y = map(lambda x:x + 2,value_list)  #map: applies a function to all the items in a list
print(list(y))
#Nested Function
def operation():  #function inside function
    def add():
        x = 1
        y = x + 1
        return y
    return add()*10
print(operation())
#Default Function
def operation(a, b=1):
    x = a + b 
    return x
print(operation(3))  #value b is not entered  -> 3+1
print(operation(3,2))   #value b is entered   -> 3+2

#Flexible Function
def operation(*args):  #multiple values can be entered
    for each in args:
        y = each + 1
        print (y)
operation(1)  # 1+1
print("")
operation(1,2)  # 1+1, 2+1
print("")
operation(1,2,3)  # 1+1, 2+1, 3+1
#Global Scope & Local Scope 

x=1    #global
def operation():
    x=2    #local
    return x

print(x)    #variable outside a function
print(operation())   #variable inside a function
#Iterators
num_list = [1,2,3]
a = iter(num_list)
print(next(a))    
print(*a)
print("")

name_list = ["ali","ayse","deniz"]
b = iter(name_list)
print(next(b))
print(*b)
print("")

word = "afyonkarahisar"
c = iter(word)
print(next(c))
print(*c)
#Zip Lists
name_list = ["ali","ayse","deniz"]
age_list = [20,30,40]
z = zip(name_list,age_list)
print(z)
print(list(z))