# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon_alopez247.csv')
data
# top 7 row of data
data.head(7)
# bottom 5 row of data
data.tail()
# data about information
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns
#number of row and column 
data.shape
# hoe many different Type_1
data.Type_1.unique()
data_unique = data.Type_1.unique()
data_unique.shape
new_type_1 = data[data.Type_1 == 'Flying']
new_type_1
x = (data['HP'] > 75)&(data['Speed'] > 100)
data [x]
data1=data.loc[:,["Attack","Speed"]]
data1.plot()
data1.plot(subplots=True)
plt.show()
# Line Plot
data.Attack.plot(kind = 'line', color = 'g',label = 'Attack',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Speed.plot(color = 'r',label = 'Speed',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')    
# Histogram
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data.Speed.plot(kind = 'hist',bins = 50)
plt.clf()
#It has 'key' and 'value'
dictionary = {'south_korea' : 'seul','russia' : 'moscow'}
print(dictionary.keys())
print(dictionary.values())
dictionary['russia'] = "st.petersburg"    # update existing entry
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['russia']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
series = data['Defense']        # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))
 # learn logic, control flow and filtering
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)
x = data['Defense']>200     # There are only one pokemons who have higher defense value than 200
data[x]
data[np.logical_and(data['Defense']>100, data['Attack']>145 )]
# gives the same result as the code in a top row
data[(data['Defense']>100) & (data['Attack']>145)]
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')
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

# For pandas we can achieve index and value
for index,value in data[['Attack']][0:1].iterrows():
    print(index," : ",value)
# import data
data=pd.read_csv('../input/pokemon_alopez247.csv')
# information of data
data.info()
#top 5 data
data.head(5)
def tuble():
    t = (2,4,6)
    return t
x,y,z = tuble()
print(x,y,z)
def tuble():
    t = (2,4,6)
    return t
x,y,z = tuble()
print(y)
x = 7 # global
def f():
    x = 9 # local
    return x
print(x)      # x = 7
print(f())    # x = 9
x = 3
def f():
    y = 2*x  # there is no local scope x
    return y
print(f())  
#give information on defined words
import builtins
dir(builtins)
# function inside function
def square():
    def add():
        x = 5
        y = 9
        z = x + y
        return z
    return add()**2
print(square())  
# default arguments
def f(p, r = 1, h = 2, n =3):
    y = p + r + h + n
    return y
print(f(5))
# We may change default arguments
print(f(5,4,3)) #p=5,r=4,h=3 and n=3(remanin same)
# flexible arguments *args
def f(*args): # *args = one or more value
    for i in args:
        print(i)      
f(1)
print("")
f(1,2,3,4)
def f(**kwargs):
    for key, value in kwargs.items():              
        print(key, " ", value)
f(Name = 'Bulbasaur', Type_2 = 'Poison', Speed = 45)
# lambda function
# faster and easier function writing
square = lambda x: x**2     # where x is name of argument
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))
#Like lambda function but it can take more than one arguments.
number_list = [1,2,3]
y = map(lambda x:x**2,number_list)
print(list(y))
# iteration example
name = "carpe diem"
it = iter(name)
print(next(it))   # print next iteration
print(next(it))
print(*it)         # print remaining iteration
# ZIP
# zip()
# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuple
print(un_list1)
print(un_list2)
print(type(un_list2)) # tuple
lst = list(un_list2)  # list
print(lst)
print(type(lst))
# Example of list comprehension       # [i + 1 for i in num1 ]: list of comprehension 
num1 = [1,2,3]                       # i+1: list comprehension syntax 
print(num1)                          # for i in num1: for loop syntax 
num2 = [i + 1 for i in num1 ]        # i: iterator 
print(num2)                          # num1: iterable object
# Conditionals on iterable
num1 = [5,10,15,0]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
threshold = sum(data.Speed)/len(data.Speed)
print('threshold = ',threshold)
data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]
data.loc[:10,["speed_level","Speed"]] 
data.head()
data.tail()
data.shape
data.info()
print(data.Type_1.value_counts(dropna =False))
data.describe()
# for example: compare attack of pokemons that are isLegendary or not
# black line at top is max
# blue line at top is 75%
# green line is median (50%)
# blue line at bottom is 25%
# black line at bottom is min
data.boxplot(column='Attack',by = 'isLegendary')
plt.show
data_n = data.head()
data_n
# melt() -> make out new data
# id_vars -> we want to keep same the column
# value_vars -> consist of variable and value
melted = pd.melt(frame=data_n,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted
#reverse of melting
melted.pivot(index = 'Name', columns = 'variable',values='value')
#concatenating two data
data1 = data.head()
data2= data.tail()
# ignore_index = True -> is provide sequential of index
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 -> vertical unite datas
conc_data_row
data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 -> horizontal unite datas
conc_data_col
data.dtypes
# convert object(str) to categorical and int to float.
data['Type_1'] = data['Type_1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
data.dtypes
# look at does pokemon data have nan value
# As you can see there are 721 entries. However Type_2 has 350 non-null object so it has 371 null object.
data.info()
# dropna= False -> we can see Type_2 has that NaN values
data["Type_2"].value_counts(dropna =False)
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type_2"].dropna(inplace = True)# inplace = True means we do not assign it to new variable. Changes automatically assigned to data
#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true
# assert 1==2 # return error because it is false
assert  data['Type_2'].notnull().all() # returns nothing because we drop nan values
data["Type_2"].fillna('empty',inplace = True)
assert  data['Type_2'].notnull().all() # returns nothing because we do not have nan values
assert data.columns[4] == 'Total'
assert data.Speed.dtypes == np.float
# data frames from dictionary
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
print(zipped)
data_dict = dict(zipped)
print(data_dict)
df = pd.DataFrame(data_dict)
df
# Add new columns
df["capital"] = ["madrid","paris"]
df
# Broadcasting
df["income"] = 0 #Broadcasting entire column
df
# Plotting all data 
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
# it is confusing
# subplots
data1.plot(subplots = True)
plt.show()
# scatter plot  
data1.plot(kind = "scatter",x="Attack",y = "Defense")
plt.show()
# hist plot  
# range -> Indicates the range of the x-axis.
# if normed is true,y axis is in the range of 0-1
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
# histogram subplot with non cumulative and cumulative
# cumulative = True -> add previous values
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
data.describe()
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
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
print(data2.loc["1992-03-10":"1993-03-16"]) # Returns data between 1992-03-10 and 1993-03-16
# find mean of columns according to year
data2.resample("A").mean()
# find mean of columns according to month
data2.resample("M").mean()
# NaN values are valued between initial and final values
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/pokemon_alopez247.csv')
data.head()
# new index column
data= data.set_index("Number")
data.head()
# indexing using square brackets
data["Total"][3]
# using column attribute and row label
data.Total[3]
# using loc accessor
data.loc[2,["Speed"]]
# Selecting only some columns
data[["HP","Attack"]]
# Difference between selecting columns: series and dataframes
print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames
# Slicing and indexing series
data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive
# Reverse slicing 
data.loc[10:1:-1,"HP":"Defense"] 
# From something to end
data.loc[1:10,"Height_m":] 
# Creating boolean series
boolean = data.HP > 200
data[boolean]
# Combining filters
first_filter = data.HP > 150
second_filter = data.Speed > 55
data[first_filter & second_filter]
# Filtering column based others
# returns the HP column of data with a speed less than 15
data.HP[data.Speed<15]
# Plain python functions
def div(n):
    return n/2
data.HP.apply(div)
# Or we can use lambda function
data.HP.apply(lambda n : n/2)
# Defining column using other columns
data["total_power"] = data.Attack + data.Defense
data.head()
# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()
# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,821,1)
data3.head()
# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/pokemon_alopez247.csv')
data.head()
# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["Type_1","Type_2"]) 
data1.head(100)
# data1.loc["Fire","Flying"] # howw to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
# pivoting: reshape tool
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
# Reverse of pivoting
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
# according to treatment take means of other features
df.groupby("treatment").mean()   # mean is aggregation / reduction method
# there are other methods like sum, std,max or min
# we can only choose one of the feature
df.groupby("treatment").age.max() 
# Or we can choose multiple features
df.groupby("treatment")[["age","response"]].min() 
df.info()