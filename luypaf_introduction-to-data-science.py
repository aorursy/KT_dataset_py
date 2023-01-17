# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output

#The part of majority of this kernel was done with help from Data ScienceTutorial for Beginners written by kanncaa1
#Thank you Sir :)

data=pd.read_csvdata=pd.read_csv('../input/gtd/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')

data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(100, 100))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(8)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.country.plot(kind = 'line', color = 'blue',label = 'country',linewidth=1,alpha = 1,grid = True,linestyle = '--')
data.region.plot(color = 'r',label = 'region',linewidth=1, alpha = 0.5,grid = True,linestyle = ':')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = iyear, y = longitude
data.plot(kind='scatter', x='iyear', y='longitude',alpha = 0.5,color = 'purple')
plt.xlabel('iyear')              # label = name of label
plt.ylabel('longitude')
plt.title('Iyear-Longitude Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data.alternative.plot(kind = 'hist',bins = 50,figsize = (10,10))
plt.show()
#create dictionary and look its keys and values
dictionary = {'chelsea' : 'lampard','manchester' : 'scholes'}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary['chelsea'] = "cole"    # update existing entry
print(dictionary)
dictionary['arsenal'] = "henry"       # Add new entry
print(dictionary)
del dictionary['chelsea']              # remove entry with key 'spain'
print(dictionary)
print('liverpool' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted


data=pd.read_csvdata=pd.read_csv('../input/gtd/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
series = data['suicide']        # data['suicide'] = series
print(type(series))
data_frame = data[['suicide']]  # data[['suicide']] = data frame
print(type(data_frame))
# Comparison operator
print(9 > 18)
print(5!=3)
# Boolean operators
print(False and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data['nkill']>1000     # There are just 4 years which have higher nkill value than 1000
data[x]
# 2 - Filtering pandas with logical_and
# There are just 3 years which have higher nkill value than 1000 and higher imonth value than 5
data[np.logical_and(data['nkill']>1000, data['imonth']>5 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['nkill']>1000) & (data['imonth']>5)]
# Stay in loop if condition( i is not equal 5) is true
i = 10
while i != 5 :
    print('i is: ',i)
    i -=1 
print(i,' is equal to 5')

lis = [6,7,8,9,10]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:6, 1:7, 2:8, 3:9, 4:10
for index, value in enumerate(lis):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'chelsea' : 'lampard','manchester' : 'scholes'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')


# For pandas we can achieve index and value
for index,value in data[['nkill']][45321:45322].iterrows():
    print(index," : ",value)
#The part of majority of this kernel was done with help from Data ScienceTutorial for Beginners written by kanncaa1
#Thank you Sir :)
# Tuble Function
#Similar to list
#The most important difference from list, which is tuple is not changed
def tuble_ex():
    """ return defined t tuble"""
    t = (2,4,6)
    return t
a,b,c = tuble_ex()
print(a,b,c)
#Scope
# guess print what
x = 5            #There was global scope
def f():
    x = 8        #There was local scope
    return x
print(x)      # x = 5 global scope
print(f())    # x = 8 local scope
# If there is not any element (x) in local scope
x = 12
def f():
    y = x/2        # there is no local scope x
    return y
print(f())         # it uses global scope x
# Firstly local scope is searched, then global scope is searched, if two of them cannot be found lastly built in scope is searched.
# If we want to learn built in scope
import builtins
dir(builtins)
#Nested function
#This function sum two variables then show square root of the sum
def square_root():              
    """ return square of value """
    def add():
        """ add two local variables """
        x = 5
        y = 4
        z = x + y
        return z
    from math import sqrt
    return sqrt(add())
print(square_root())    
# Default arguments
#b and c are default values in this sample
def f(a, b = 4, c = 2):
    y = a * b * c
    return y
print(f(5))
# what if we want to change all default arguments
print(f(5,4,3))   
#what if we want to change just a default argument and another default argument is not changed
print(f(5,8))     
# flexible arguments *args
#We can use if we don't know number of list elements
def f(*args):
    for i in args:
        print(i)
f(5)
print("")
f(1,2,3,4)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():             
        print(key, " ", value)
f(team = 'chelsea', captain = 'lampard', fee = 534354)
# Lambda Function
#Maybe you can want to write faster code :)
sub = lambda x,y: x-y     # where x,y are names of arguments
print(sub(4,8))
delta = lambda a,b,c: b*b-4*a*c   # where a,b,c are names of arguments
print(delta(1,2,3))
#Anonymous Function
#Maybe you want to print one more than arguments
number_list = [1,2,3]
y = map(lambda x:x*5,number_list)
print(list(y))
# iteration example
#Iterator use if you seperate list elements
name = "abracadabra"
it = iter(name)
print(next(it))    # print next iteration
print(*it)         # print remaining iteration
# zip example
#You want to combine two lists
list1 = [1,2,3,4]
list2 = ["a","b","c","d"]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
#unzip example
#You want to seperate two combining lists
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))
# list comprehension
# you can make what do you want operation to all list elements using just a for loop
num1 = [1,2,3]
num2 = [i * 2 for i in num1 ]
print(num2)
# Conditionals on iterable
#Also you can use conditional statement with list comprehension
num1 = [5,10,15]
num2 = [i**2 if i == 5 else i-5 if i > 10 else i+5 for i in num1]
print(num2)
# lets return Global Terrorism Database csv and make one more list comprehension example
threshold = sum(data.success)/len(data.success)
data["success_level"] = ["high" if i > threshold else "low" for i in data.success]
data.loc[1000:1010,["success_level","success"]] 
data=pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
data.head()  # head shows first 5 rows
data.tail() #Shows last 5 rows
# columns gives column names of features
data.columns
# shape gives number of rows and columns in a tuble
data.shape
data.info()

# For example lets look frequency of city of athletes
print(data['City'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 22426 London or 12977 Barcelona
#Dtatistical infos
# For example min Age is 10 or max weight is 214
data.describe() #ignore null entries
# For example: compare Age of athletes that are which Medal
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Age',by = 'Medal')
#Tidy data
# Firstly create new data from Athletes data to explain melt nore easily.
data_new = data.head()    # Just 5 rows
data_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Team','Sport'])
melted
#Pivoting data is that reverse melting
# Index is name
# Make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')
#Concatenating Data
#You can concate two dataframes
# Firstly lets create 2 data frames
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
#Also you can concate in columns
data1 = data['Games'].head()
data2= data['Height'].head()
conc_data_col = pd.concat([data1,data2],axis =1)
conc_data_col
#You want to show datatypes
data.dtypes
# Sometimes your datas have any nan values
# As you can see there are 271116 entries. However Age has 261642 non-null object so it has 9474 null object.
data.info()
# Lets check Age
data["Age"].value_counts(dropna =False)
# As you can see, there are 9474 NAN value
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Age"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?
# What is the assert statement:
# If you want to control any statement for true or false you can use assert keywords
assert 1==1 # return nothing because it is true
# assert 1==2 # return error because it is false
assert  data['Age'].notnull().all() # returns nothing because we drop nan values
data["Age"].fillna('empty',inplace = True) #Maybe you want to write different word (like empty) instead Nan
assert  data['Age'].notnull().all() # returns nothing because we do not have nan values
 
# With assert statement we can check a lot of things such as assert data.columns[1] == 'Name'
# assert data.columns[1] == 'Name'
# assert data.Age.dtypes == np.int
# data frames from dictionary
team = ["Barcelona","Juventus"]
price = ["50","10"]
list_label = ["team","price"]
list_col = [team,price]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
# Maybe you want to add new columns
df["captain"] = ["Messi","Pirlo"]
df

# Broadcasting
df["stad_capacity"] = 0 #Broadcasting entire column(like default value )
df

data=pd.read_csv("../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv")
data.info()
# Plotting all data 
data1 = data.loc[:,["NA_Sales","Critic_Score","Other_Sales"]]
data1.plot()
#Seems confusing
# subplots
data1.plot(subplots = True)
plt.show()
# scatter plot  
data1.plot(kind = "scatter",x="NA_Sales",y = "Critic_Score")
plt.show()
# histogram plot  
data1.plot(kind = "hist",y = "Critic_Score",bins = 50,range= (15,100),normed = True)
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Critic_Score",bins = 50,range= (15,100),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Critic_Score",bins = 50,range= (15,100),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
#Time series
time_list = ["1998-02-10","2001-05-19"]
print(type(time_list[1])) # As you can see date is string
datetime_object = pd.to_datetime(time_list) #And then convert to object
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of Video Games data and add it a time list
data2 = data.head()
date_list = ["1998-02-10","1998-08-22","1999-03-04","2000-11-15","2001-03-06"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2= data2.set_index("date")
data2 
# Now we can select according to our date index
print(data2.loc["2001-03-06"])
print(data2.loc["1998-03-10":"2001-03-06"])
#Resampling
# You can use data2 that we create at previous part
data2.resample("A").mean()
# Lets resample with month
data2.resample("M").mean()
# As you can see there are a lot of nan because data2 does not include all months
# You can solve this problem with interpolate
# You can interpolete from first value
data2.resample("M").first().interpolate("linear")
# Maybe you want to interpolate with mean()
data2.resample("M").mean().interpolate("linear")
# read data
data = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
data.head()
#Set index ID
data = data.set_index("ID")
data.head()
# indexing using square brackets
data["Age"][1]

# Another way using column attribute and row label
data.Age[1]
# If you want to view more details
data.loc[1,["Age"]]
# Maybe you need to more than one column
data[["Age","Games"]]
# Difference between selecting columns: series are one dimension and dataframes are two dimension
print(type(data["Age"]))     # series
print(type(data[["Age"]]))   # data frames
# Slicing and indexing series
# You want to display values in a certain range.
data.loc[1:10,"Age":"City"]   # Views from age to City (city is interval because of pandas library)
# If you want to reverse your slicing
data.loc[10:1:-1,"Age":"City"] 
# From what do you want to end of your dataframe
data.loc[1:10,"Year":] 
# Creating filtering series
filtering = data.Height > 220
data[filtering]
# Combining filters
first_filter = data.Height > 220
second_filter = data.Weight > 130
data[first_filter & second_filter]
# Filtering your selected column and views from same data of another column
data.Age[data.Height>225]
# You can use any function all datas of selected column 
def mul(n):
    return n*2
data.Age.apply(mul)
# Same operation with lambda function
data.Age.apply(lambda n : n*2)
# Also you create a column using other columns
data["index_body"] = data.Weight / data.Height**2
data.head()
# our index name is this:
print(data.index.name)
# You want to change it
data.index.name = "#"
data.head()
# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3 = data.copy()
# If you want to start index from 500
data3.index = range(500,271616,1)
data3.head()
# Setting index : Games is outer City is inner index
data1 = data.set_index(["Games","City"]) 
data1.head(100)
dic = {"disease":["A","A","B","B"],"gender":["F","M","F","M"],"probability":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
# pivoting 
df.pivot(index="disease",columns = "gender",values="probability")
df1 = df.set_index(["disease","gender"])
df1
# We already learn this before 
# What's unstack
df1.unstack(level=0)
# Disease and gender are columns anymore and filtering as disease
df1.unstack(level=1)
# Disease and gender are columns anymore and filtering as gender
# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2
# Melting
pd.melt(df,id_vars="disease",value_vars=["age","probability"])
# Also we show any statistical values of datas.
df.groupby("disease").std()   # std() method
# there are other methods like sum, mean,max or min
# You want to display selected feature of another selected feature
df.groupby("disease").age.max() 
# Or you can choose multiple features
df.groupby("disease")[["age","probability"]].min() 
df.info()
