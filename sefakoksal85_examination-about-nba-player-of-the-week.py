# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/NBA_player_of_the_week.csv")
data.info()
data.head()
data.tail()
data.corr()
f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt= '.1f', ax=ax)
plt.show()
data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns]
print(data.columns)
data.columns = [ each.lower() for each in data.columns]
print(data.columns)
data.age.plot(kind="line", linestyle=":", color="r", grid=True, linewidth=1)
plt.title("age")
plt.show()
data.plot(kind='scatter', x='age', y='real_value',alpha = 0.5,color = 'red')
plt.xlabel('age')              # label = name of label
plt.ylabel('real_value')
plt.title('age_real_value_comparison')
data.age.plot(kind = 'hist',bins = 50,figsize = (12,12), grid=True)
plt.show()
# USER DEFINED FUNCTION
def example():
    t = (1,2,3)
    return t
a,b,c = example()
print(a,b,c)
# scope
x = 3
def f():
    x = 5
    return x
print(x)      # x = 3 global scope
print(f())    # x = 5 local scope
# What if there is no local scope
x = 3
def f():
    y = 9*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
#nested function 
def example():
    def other():
        x = 3
        y = 5
        z = x + y
        return z
    return other()**2
print(example())    
# default arguments
def f(a, b = 2, c = 3):
    y = a + b + c
    return y
print(f(5))
# what if we want to change default arguments
print(f(5,4,3)) 
# flexible arguments *args
def f(*args):
    for i in args:
        print(i)
f(3)
print("")
f(1,2,3,4)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    for key, value in kwargs.items():               
        print(key, " ", value)
f(country = 'turkey', capital = 'ankara', population = 6000000)
# lambda function
square = lambda x: x**2     # where x is name of argument
print(square(5))
total = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(total(1,2,3))
#ANONYMOUS FUNCTION
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
print(type(list(un_list1)))
# Conditionals on iterable
num1 = [5,10,15,3,8,9]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)

a = ["small" if x < 9 else "equal" if x==9 else "big" for x in num1] 
print(a)
threshold = sum(data.age)/len(data.age)
print(threshold)
threshold = sum(data.age)/len(data.age)
data["age_condition"] = ["high" if i > threshold else "low" for i in data.age]
data.loc[:10,["age_condition","age"]] 
data.head()
data.tail()
data.info()
data.columns
data.shape
print(data['age'].value_counts(dropna =False))
print(data['conference'].value_counts(dropna =False))
data.describe()
data.boxplot(column='age',by = 'seasons_in')
data_new = data.head()  
data_new
melted = pd.melt(frame=data_new,id_vars = 'player', value_vars= ['age','weight'])
melted
melted.pivot(index = 'player', columns = 'variable',values='value')
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) 
conc_data_row
data1 = data['player'].head()
data2 = data['age'].head()
data3= data['weight'].head()
conc_data_col = pd.concat([data1,data2,data3],axis =1) 
conc_data_col
data.dtypes
# DATA TYPES
data['position'] = data['position'].astype('category')
data['season'] = data['season'].astype('float')
data.dtypes
data.info()
data["conference"].value_counts(dropna =False)
data1=data   
data1["conference"].dropna(inplace = True) 
assert  data['conference'].notnull().all() # returns nothing because we drop nan values
data["conference"].value_counts(dropna =False) #new version of data["conference"] as you see no NaN value in there
data["conference"].fillna('empty',inplace = True) 