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
data=pd.read_csv('../input/data.csv')
data.info() # this code give us general information about our data
data.describe() #this code give us  digital information. After we will look closely. 
data.head() # To look first 5 values

#data.tail() # to look last 5 values
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Finishing.plot(kind = 'line', color = 'g',label = 'Finishing',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Positioning.plot(color = 'r',label = 'Positioning',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# x = Special, y = Skill Moves
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.plot(kind='scatter', x='Special', y='Skill Moves',alpha = 0.5,color = 'red')
plt.xlabel('Special')              # label = name of label
plt.ylabel('Skill Moves')
plt.title('Special Skill Moves Scatter Plot')  # title = title of plot
plt.show()          # for showing plot
# bins = number of bar in figure
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.SprintSpeed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data.SprintSpeed.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()
#create dictionary and look its keys and values
dictionary = {'istanbul' : 'besiktas','paris' : 'psg'}
print("dictionary:",dictionary)
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['istanbul'] = "galatasaray"    # update existing entry
print(dictionary)
dictionary['madrid'] = "realmadrid"       # Add new entry
print(dictionary)

del dictionary['istanbul']              # remove entry with key 'spain'
print(dictionary)
print('paris' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)

# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
data=pd.read_csv('../input/data.csv')
series = data['Club']        # data['Club'] = series
print(type(series))
data_frame = data[['Agility']]  # data[['Agility']] = data frame
print(type(data_frame))

# Comparison operator
print(21 > 2)
print(3!=2) #It means that 3 is not equal to 2
# Boolean operators
print(True and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data['Age']>43     # There are only 3 footballers who have older than 43
data[x]
# 2 - Filtering pandas with logical_and
# There are only 2 footballers who are older than 43 and smaller Potential value than 60
data[np.logical_and(data['Age']>43, data['Potential']<60 )]
data.columns  # now we can look every columns' names
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['Age']>43) & (data['Potential']<60) ]
# Stay in loop if condition( i is not equal 5) is true
a = 0
while a != 5 :
    print('a is: ',a)
    a +=1 
print(a,' is equal to 5')
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
dictionary = {'spain':'barcelona','france':'psg'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['Value']][0:6].iterrows():
    print(index," : ",value)


# example of what we learn above
def fake_func():
    """ return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = fake_func()
print(a,b,c)
# guess print what
x = 2
def f():
    x = 3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope
# What if there is no local scope
x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
# First local scope searched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope
import builtins
dir(builtins)
#nested function
def square():
    """ return square of value """
    def add():
        """ add two local variable """
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
f(city = 'madrid', team = 'realmadrid', value = "€128M")
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
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)
# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
# lets return our data csv and make one more list comprehension example
# lets classify footballers whether they have high or low potential. Our threshold is average potential.
threshold = sum(data.Potential)/len(data.Potential)
print(threshold)
data["pot_level"] = ["high" if i > threshold else "low" for i in data.Potential]
data.loc[:10,["pot_level","Potential"]] # we will learn loc more detailed later
data=pd.read_csv('../input/data.csv')
data.head(10) # first 10 player
#data.tail()
#data.shape()  # shape gives number of rows and columns in a tuble our data is 18207X89
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()
# For example lets look frequency of players Nationality
print(data['Nationality'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 1662 player from England,1198 player from Germany etc...
# for example mean of age is 25.12
data.describe() #ignore null entries
# For example: compare Potential of players that are real face  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Potential',by = 'Real Face')
# Firstly I create new data from fifa data to explain melt more easily.
data_new = data.head()    # I only take 5 rows into new data
data_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Nationality','Club'])
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
data1 = data['Nationality'].head()
data2= data['Value'].head()
data3= data['Name'].head()
conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col
data.dtypes.head(10)
# lets convert object(str) to categorical and int to float.
data['Nationality'] = data['Nationality'].astype('category')
data['Age'] = data['Age'].astype('float')
# As you can see Nationality is converted from object to categorical
# And Age ,s converted from int to float
data.dtypes.head(10)
# Lets look at does player data have nan value?
# As you can see there are 18207  entries. However Club has 17966 non-null object so it has 241 null object.
data.info()
# Lets check club
data["Club"].value_counts(dropna =False)
# As you can see, there are 241 NAN value
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Club"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?
#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment
# assert 1==2 # return error because it is false
assert  data['Club'].notnull().all() # returns nothing because we drop nan values
data["Club"].fillna('empty',inplace = True)
assert  data['Club'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example
# assert data.columns[1] == 'Name'
# assert data.Speed.dtypes == np.int
# we dont need player Id photo and flag etc. lets drop them all
data=data.drop(['ID','Photo','Flag'],axis=1)
data.head()
threshold1 = data.Dribbling.mean()
print(threshold1)
data["fin_level"] = ["high" if i > threshold1 else "low" for i in data.Finishing]
data.loc[:10,["fin_level","Finishing"]]
#lets find all the turkish player
turk= data.Nationality == 'Turkey'
data[turk]

# data frames from dictionary
country = ["Spain","France"]
team = ["realmadrid","psg"]
list_label = ["country","team"]
list_col = [country,team]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
# Add new columns
df["capital"] = ["madrid","paris"]
df
# Broadcasting
df["value"] = 0 #Broadcasting entire column
df
# Plotting all data 
data1 = data.loc[:,["Age","Potential","Overall"]]
data1.plot()
# it is confusing
# subplots
data1.plot(subplots = True)
plt.show()
# scatter plot  
data1.plot(kind = "scatter",x="Potential",y = "Overall")
plt.show()
# hist plot  
data1.plot(kind = "hist",y = "Age",bins = 50,range= (0,50),normed = True)
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Age",bins = 50,range= (0,50),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Age",bins = 50,range= (0,50),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.describe()
# read data
data = pd.read_csv('../input/data.csv')
data= data.set_index("Unnamed: 0")
data.head()
# indexing using square brackets
data["Nationality"][1]
# using column attribute and row label
data.Nationality[1]
# using loc accessor
data.loc[1,["Nationality"]]
# Selecting only some columns
data[["Name","Nationality"]]
# Difference between selecting columns: series and dataframes
print(type(data["Nationality"]))     # series
print(type(data[["Nationality"]]))   # data frames
# Slicing and indexing series
data.loc[1:10,"Name":"Potential"]   # 10 and "Potential" are inclusive
# Reverse slicing 
data.loc[10:1:-1,"Name":"Potential"] 
# From something to end
data.loc[1:10,"SlidingTackle":] 
# Creating boolean series
boolean = data.Club == 'Juventus'
data[boolean]
# Combining filters
first_filter =  data.Club == 'Juventus'
second_filter = data.Potential > 88
data[first_filter & second_filter]
# Filtering column based others
data.Age[data.Overall>90]
# Plain python functions
def div(n):
    return n/2
data.Overall.apply(div)
# Or we can use lambda function
data.Overall.apply(lambda n : n/2)
# Defining column using other columns
data["total_power"] = data.Overall + data.Potential
data.head()
#at the end of the columns our total_power
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
# lets make index start from 300. It is not remarkable change but it is just example
data3.index = range(300,18507,1)
data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# It was like this
# data= data.set_index("Unnamed: 0")
# also you can use 
# data.index = data["Unnamed: 0"]
# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/data.csv')
data.head()
# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["Name","Age"]) 
data1.head(100)
# data1.loc["Nationality","Club"] # how to use indexes
dic = {"teams":["A","A","B","B"],"players":["cris","thomas","bale","mbap"],"response":[10,45,5,9],"age":[22,23,27,19]}
df = pd.DataFrame(dic)
df
# pivoting
df.pivot(index="teams",columns = "players",values="response")
df1 = df.set_index(["teams","players"])
df1
# lets unstack it
# level determines indexes
df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2
df
# df.pivot(index="teams",columns = "players",values="response")
pd.melt(df,id_vars="teams",value_vars=["age","response"])
# We will use df
df
# according to teams take means of other features
df.groupby("teams").mean()   # mean is aggregation / reduction method
# there are other methods like sum, std,max or min
# we can only choose one of the feature
df.groupby("teams").age.max() 
# Or we can choose multiple features
df.groupby("teams")[["age","response"]].min() 
df.info()
# as you can see gender is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
#df["gender"] = df["gender"].astype("category")
#df["treatment"] = df["treatment"].astype("category")
#df.info()