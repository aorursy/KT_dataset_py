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

# close warning
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns",None) 
pd.set_option("display.max_rows",None)
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2015.csv")
data.info()# Display the content of data
data.rename(columns={"Economy (GDP per Capita)":"economy","Health (Life Expectancy)":"health","Trust (Government Corruption)":"Trust"}, inplace=True)
# shape gives number of rows and columns in a tuple
data.shape
data.columns
data.columns = [each.replace(" ","_") if(len(each.split())>1) else each for each in data.columns]
print(data.columns)
data.columns = [each.lower() for each in data.columns]
print(data.columns)
data.describe()
data.head()
data.tail()
data.sample(5)
data.dtypes
# Display positive and negative correlation between columns
data.corr()
#sorts all correlations with ascending sort.
data.corr().unstack().sort_values().drop_duplicates()
#correlation map
plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidth=".5", cmap="YlGnBu", fmt=".2f")
plt.show()
#figsize - image size
#data.corr() - Display positive and negative correlation between columns
#annot=True -shows correlation rates
#linewidths - determines the thickness of the lines in between
#cmap - determines the color tones we will use
#fmt - determines precision(Number of digits after 0)
#if the correlation between the two columns is close to 1 or 1, the correlation between the two columns has a positive ratio.
#if the correlation between the two columns is close to -1 or -1, the correlation between the two columns has a negative ratio.
#If it is close to 0 or 0 there is no relationship between them.
data.isnull().head(15)
data.isnull().sum() #Indicates values not defined in our data
data.isnull().sum().sum()  #Indicates sum of values in our data
data[["happiness_score"]].isnull().head(15)
data.sort_values("happiness_score", ascending=False).head(10)
data.sort_values("happiness_score", ascending=True).head(10)
data[["happiness_score","economy","family","health"]].head(10)
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.happiness_score.plot(kind="line", color="g", label="happiness_score", linewidth=1, alpha=0.5, grid=True, figsize=(12,12))
data.economy.plot(kind="line", color="r", label="economy", linewidth=1, alpha=0.5, grid=True)
data.family.plot(kind="line", color="y", label="family", linewidth=1, alpha=0.5, grid=True)
data.health.plot(kind="line", color="b", label="health", linewidth=1, alpha=0.5, grid=True)
plt.legend(loc="upper right")# legend = puts label into plot
plt.xlabel("x axis")         # label = name of label
plt.ylabel("y axis")
plt.title("line Plot")       # title = title of plot
plt.show()

#plt.xticks(np.arange(first value,last value,step)) 
#plt.xticks(np.arange(0,800,30)) #Determines the ranges of values in the x-axis
#plt.yticks(np.arange(0,300,30)) #Determines the ranges of values in the y-axis
#plt.show()
# subplots
data.plot(subplots = True, figsize=(12,12))
plt.show()
plt.subplot(4,2,1)
data.family.plot(kind="line", color="orange", label="family", linewidth=1, alpha=0.5, grid=True, figsize=(10,10))
data.happiness_score.plot(kind="line", color="green", label="family", linewidth=1, alpha=0.5, grid=True, figsize=(10,10))
plt.ylabel("family")
plt.subplot(4,2,2)
data.generosity.plot(kind="line", color="blue", label="generosity", linewidth=1, alpha=0.5, grid=True, linestyle=":")
plt.ylabel("generosity")
plt.subplot(4,2,3)
data.trust.plot(kind="line", color="green", label="trust", linewidth=1, alpha=0.5, grid=True, linestyle="-.")
plt.ylabel("trust")
plt.subplot(4,2,4)
data.freedom.plot(kind="line", color="red", label="freedom", linewidth=1, alpha=0.5, grid=True)
plt.ylabel("freedom")
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind="scatter", x="happiness_score", y="economy", alpha=0.5, color="green", figsize=(5,5))
plt.xlabel("happiness_score")    # label = name of label
plt.ylabel("economy")
plt.title("Happiness Score Economy Scatter Plot") # title = title of plot
plt.show()
data.plot(kind="scatter", x="economy", y="health", alpha=0.5, color="blue", figsize=(5,5))
plt.xlabel("economy")    # label = name of label
plt.ylabel("health")
plt.title("Economy Health Scatter Plot") # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.happiness_score.plot(kind="hist",color="orange", bins=160, figsize=(10,10))
plt.show()
data.happiness_score.head(30).plot(kind="bar")
plt.show()
data.happiness_score.sample(30).plot(kind="bar")
plt.show()
data.happiness_score.head(100).plot(kind="area")
plt.show()
# clf() = cleans it up again you can start a fresh
data.happiness_score.plot(kind="hist", bins=50)
plt.clf() # We can not see plot if we use clf() method
plt.show()
#we dont use.its just example
dic2 = [{"id": 825, "name": "Orhan"}, {"id": 851, "name": "Kadir"},{"id": 856, "name": "Cemal"}]
df2 = pd.DataFrame(dic2)
df2
#create dictionary and look its keys and values
dictionary = {"Turkey":"Ankara","Germany":"Berlin"}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique
dictionary["Turkey"] = "Ankara" # update existing entry
print(dictionary)
dictionary["France"] = "Paris"    #Add new entry
print(dictionary)
del dictionary["France"]           # remove entry with key 'spain'
print(dictionary)
print("France" in dictionary)     # check include or not
dictionary.clear()                # remove all entries in dict
print(dictionary)
# In order to run all code you need to take comment this line
#del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
print(type(data)) # pandas.core.frame.DataFrame
print(type(data[["freedom"]])) #pandas.core.frame.DataFrame
print(type(data["freedom"])) #pandas.core.series.Series
print(type(data["freedom"].values)) #numpy.ndarray
series = data['freedom']        # data['Defense'] = series
data_frame = data[['freedom']]  # data[['Defense']] = data frame

print(type(series))
print(type(data_frame))

print(series.head(10))
data_frame.head(10)
# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data["happiness_score"]>5.0
data[x]
# 2 - Filtering pandas with logical_and
data[np.logical_and(data["family"]>1.3,data["economy"]>1.3)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data["family"]>1.3) & (data["economy"]>1.3)]
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5:
    print("i is: ",i)
    i+=1
print(i," is equal to 5")
# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]

for i in lis:
    print("i is: ",i)
print("")    
# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index,value in enumerate(lis):
    print(index," : ",value)
print("")
# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = dictionary = {'Turkey':'Ankara','France':'Paris'}
for key in dictionary:
    print(key)
print("")
for key,value in dictionary.items():
    print(key," : ",value)
print("")
# For pandas we can achieve index and value
for index,value in data[["freedom"]][0:5].iterrows():
    print(index," : ",value)
data[["freedom"]][0:5]
# example of what we learn above
def tuple_ex():
    """ return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuple_ex()
print(a,b,c)
# guess print what
x = 2
def f():
    x=3
    return x
print(x)      # x = 2 global scope
print(f())    # x = 3 local scope
# What if there is no local scope
x = 5
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
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
f(1,1,2)
print("")
f(1,2,3,4)
print("")
f("orhan","kadir","cemal",1)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():     # If you do not understand this part turn for loop part and look at dictionary in for loop
        print(key, " ", value)
f(country = 'Turkey', capital = 'Ankara', population = 80000000)
# lambda function
square = lambda x: x**2     # where x is name of argument
print(square(4))
tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments
print(tot(1,2,3))
number_list=(1,2,3,4,5,6,7,8,9)
y = map(lambda x : x**2,number_list)
#liste_Y = list(y) 
#print(liste_Y) #[1, 4, 9, 16, 25, 36, 49, 64, 81]
#OR short way
print(list(y)) #[1, 4, 9, 16, 25, 36, 49, 64, 81]
# iteration example
name = "Orhan"
itr = iter(name)
print(next(itr))# print next iteration
print(next(itr))# print next iteration
print(*itr)     # print remaining iteration
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)  #converting zip to list type
print(z_list)
print("")    
itr = iter(z_list) 
print(next(itr))   # print next iteration
print(*itr)        # print remaining iteration
un_zip = zip(*z_list)
unlist1,unlist2 = list(un_zip) # unzip returns tuple
print(unlist1)
print(unlist2)
print(type(unlist1))
print(type(list(unlist1))) #if we want to change data type tuple to list we need to use list() method.
num1 = [1,2,3]
num2 = [i+1 for i in num1]
print(num2)
#OR
print([i+1 for i in num1])
# Conditionals on iterable
num1 = [5,10,15]
num2 = [i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]
print(num2)
# lets return 2015.csv and make one more list comprehension example
# lets classify happiness_score whether they have high or low. Our threshold is happiness_score.
threshold = sum(data.happiness_score)/len(data.happiness_score)
data["happiness_score_level"] = ["high" if i>threshold else "low" for i in data.happiness_score]
data.loc[60:90,["happiness_score_level","happiness_score"]]
data = pd.read_csv('../input/2015.csv')
data.head()  # head shows first 5 rows
data.rename(columns={"Economy (GDP per Capita)":"economy","Health (Life Expectancy)":"health","Trust (Government Corruption)":"Trust"}, inplace=True)

data.columns = [each.replace(" ","_") if(len(each.split())>1) else each for each in data.columns]
print(data.columns)

data.columns = [each.lower() for each in data.columns]
print(data.columns)
# tail shows last 5 rows
data.tail()
# columns gives column names of features
data.columns
# shape gives number of rows and columns in a tuple
data.shape
data.dtypes
data["region"].unique() #shows the unique region values
(data["happiness_score"] > 1).head(20) # We can filter the data if we want 
data["happiness_score"] > 1 # We can filter the data if we want 
data[data["happiness_score"] > 1].head(20)
# For example lets look frequency of region types
print(data["region"].value_counts(dropna=False,sort=True))# if there are nan values that also be counted
#sort : boolean, default True   =>Sort by values
#dropna : boolean, default True =>Don’t include counts of NaN.
# As it can be seen below there are 40 Sub-Saharan Africa region or 29 Central and Eastern Europe region
# As you can see, there are no NAN values
# For example: compare happiness_score of region
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
#Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1).     (Q3-Q1) = IQR
data.boxplot(column='happiness_score',by = 'region',fontsize=9,figsize=(20,20))

data2 = data[data["region"]=="Western Europe"]
print(data2.happiness_score.max())
print(data2.happiness_score.quantile(q=0.75))
print(data2.happiness_score.quantile(q=0.5))
print(data2.happiness_score.quantile(q=0.25))
print(data2.happiness_score.min())

data3 = data[data["region"]=="North America"]
print(data3.happiness_score.max())
print(data3.happiness_score.quantile(q=0.75))
print(data3.happiness_score.quantile(q=0.5))
print(data3.happiness_score.quantile(q=0.25))
print(data3.happiness_score.min())

data4 = data[data["region"]=="Australia and New Zealand"]
print(data4.happiness_score.max())
print(data4.happiness_score.quantile(q=0.75))
print(data4.happiness_score.quantile(q=0.5))
print(data4.happiness_score.quantile(q=0.25))
print(data4.happiness_score.min())

data5 = data[data["region"]=="Latin America and Caribbean"]
print(data5.happiness_score.max())
print(data5.happiness_score.quantile(q=0.75))
print(data5.happiness_score.quantile(q=0.5))
print(data5.happiness_score.quantile(q=0.25))
print(data5.happiness_score.min())
# FINDING OUTLIERS
#we should change the formula according to our data

#FOR data2
print("max outliers=",[x for x in data2.happiness_score if x>(data2.happiness_score.quantile(0.75)+1.5*(data2.happiness_score.quantile(0.75)-data2.happiness_score.quantile(0.25)))])
print("min outliers=",[x for x in data2.happiness_score if x<(data2.happiness_score.quantile(0.25)-1.5*(data2.happiness_score.quantile(0.75)-data2.happiness_score.quantile(0.25)))])
print("")

#FOR data3
print("max outliers=",[x for x in data3.happiness_score if x>(data3.happiness_score.quantile(0.75)+1.5*(data3.happiness_score.quantile(0.75)-data3.happiness_score.quantile(0.25)))])
print("min outliers=",[x for x in data3.happiness_score if x<(data3.happiness_score.quantile(0.25)-1.5*(data3.happiness_score.quantile(0.75)-data3.happiness_score.quantile(0.25)))])
print("")

#FOR data4
print("max outliers=",[x for x in data4.happiness_score if x>(data4.happiness_score.quantile(0.75)+1.5*(data4.happiness_score.quantile(0.75)-data4.happiness_score.quantile(0.25)))])
print("min outliers=",[x for x in data4.happiness_score if x<(data4.happiness_score.quantile(0.25)-1.5*(data4.happiness_score.quantile(0.75)-data4.happiness_score.quantile(0.25)))])
print("")

#doing with for loop
#FOR data5
for x in data5.happiness_score:
    if x>(data5.happiness_score.quantile(0.75)+1.5*(data5.happiness_score.quantile(0.75)-data5.happiness_score.quantile(0.25))):
       print("max outliers=",x)
    elif x<(data5.happiness_score.quantile(0.25)-1.5*(data5.happiness_score.quantile(0.75)-data5.happiness_score.quantile(0.25))):
       print("min outliers=",x)
# Firstly I create new data from 2015 data to explain melt more easily.
data_new = data.head(5)    # I only take 5 rows into new data
data_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new, id_vars = "country",value_vars=["economy","health"])
melted
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index="country", columns = "variable", values="value")
# Firstly lets create 2 data frame
data1 = data.head()
data2 = data.tail()
v_concat = pd.concat([data1,data2],axis=0,ignore_index=True)# axis = 0 : adds dataframe
v_concat
data1 = data.country.head()
data2 = data.happiness_score.head()
h_concat = pd.concat([data1,data2], axis=1)
h_concat
data.info()
data1 = data.country.head(10)
data2 = data.happiness_score.head(10)
data3 = data.trust.head(10)
data4 = data.region.head(10)
h_concat = pd.concat([data4+" - "+data1,data2,data3], axis=1)
h_concat
data.dtypes
#lets convert object(str) to categorical and float to int.
#DONT forget ,Setting return back default setting to int
#data["region"] = data["region"].astype("category")
#data.freedom = data.freedom.astype("int")
#data.freedom[0:10] #as you see it is converted from int to float
# As you can see region is converted from object to categorical
# And freedom is converted from float to int
data.dtypes
data = pd.read_csv("../input/2015.csv")
data.rename(columns={"Economy (GDP per Capita)":"economy","Health (Life Expectancy)":"health","Trust (Government Corruption)":"Trust"}, inplace=True)
data.columns = [each.replace(" ","_") if(len(each.split())>1) else each for each in data.columns]
data.columns = [each.lower() for each in data.columns]
print(data.columns)

data2 = data.copy(deep=True)
data2["region"][4:8] = np.nan
data2
# Lets chech happiness_score
data2["region"].value_counts(dropna =False)
# As you can see, there are 4 NAN value
# Lets drop nan values
# also we will use data to fill missing value
data2["region"].dropna(inplace = True)
data2
#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true
#In order to run all code, we need to make this line comment
#assert 1==2 # return error because it is false
assert  data2['region'].notnull().all() # returns nothing because we drop nan values
data2["region"].fillna("empty",inplace = True)
data2 
#you can not assign empty values after delete nan values
#if you want to assign empty values firstly make that! after import csv file
# # With assert statement we can check a lot of thing. For example
assert data2.columns[1] == "region"
assert data2.happiness_score.dtype == "float"
#OR
assert data2.region.dtype == "object"
assert data2.happiness_score.dtype == "float64"
print(data2.happiness_score.dtypes)
country = ["Turkey","France"]
population = ["1000","2000"]
list_label = ["country","population"]
list_col = [country,population]
print(list_col)
zipped = list(zip(list_label,list_col))
print(zipped)
data_dict = dict(zipped)
print(data_dict)
df = pd.DataFrame(data_dict)
df
df["capital"]=["madrid","paris"]
df
df["income"] = 0
df
# Plotting all data 
data1 = data.loc[:,["happiness_score","freedom","health"]]
data1.plot()
# SAME THING
#data.happiness_score.plot()
#data.freedom.plot()
#data.health.plot()
# subplots
data1.plot(subplots = True)
plt.show()
# scatter plot  
data1.plot(kind = "scatter",x="freedom",y = "health")
plt.show()
# hist plot  
data1.happiness_score.plot(kind ="hist",range= (0,10),bins=50)
plt.show()
# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "happiness_score",color="orange",bins = 50,range= (0,10),ax = axes[0])
data1.plot(kind = "hist",y = "happiness_score",color="green",bins = 50,range= (0,10),ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
# In order to practice lets take head of 2015.csv data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
#OR
#data2.set_index("date",inplace=True)
data2 
#bütün columnları ve rowları gösterir.
pd.set_option("display.max_columns",None) 
pd.set_option("display.max_rows",None)

# Now we can select according to our date index
print(data2.loc["1993-03-16"]) #print(data2.loc["1993-03-16",:]) same thing
print(data2.loc["1992-03-10":"1993-03-16"])
# We will use data2 that we create at previous part
data2.resample("A").mean() #yıldan yıla featureların kendi içinde ortalaması
# Lets resample with month
data2.resample("M").mean()
# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()
data2.resample("M").mean().interpolate("linear")
data1 = data.head(10)
data1
# indexing using square brackets
data1["happiness_rank"][1]
# using column attribute and row label
data1.happiness_rank[1]
# using loc accessor
data1.loc[2,["happiness_rank"]]
# Selecting only some columns
data1[["happiness_rank"]]
# Difference between selecting columns: series and dataframes
print(type(data["freedom"]))     # series
print(type(data[["freedom"]]))   # data frames
# Slicing and indexing series
data.loc[1:10,"health":"generosity"]   # 10 and "Defense" are inclusive
# Reverse slicing 
a =data.loc[10:1:-1,"generosity":"health":-1] 
a
# From something to end
data.loc[1:10,"trust":] 
# Creating boolean series
boolean = data.health > 0.95
data[boolean]
# Combining filters
first_filter = data.family > .95
second_filter = data.health > .95
data[np.logical_and(first_filter,second_filter)]
#OR
#data[np.logical_and(first_filter,second_filter)]
# Filtering column based others
data.country[data.happiness_score>7]
# Filtering column based others
data[["freedom"]][data.happiness_score>7]
# Filtering column based others
a = data[data.happiness_score>7]
a[["trust"]]
# Plain python functions
def div(n):
    return n/2
data["new_happiness_score"]=data.happiness_score.apply(div)
data
data["new_happiness_score"] = data.happiness_score.apply(lambda hp : hp/2)
data
# Defining column using other columns
data["new_total_happiness_score"] = data.trust + data.economy
data.head()
# our index name is this:
print(data.index.name)
#lets change it
data.index.name = "index_name"
data.head()
# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index
data2 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just example
data2.index = range(100,258,1)#100 exclusive->258
data2.tail()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# It was like this
# data= data.set_index("happiness_rank")
# also you can use 
data.index = data["happiness_rank"]
data.index = data["freedom"]
data.index = data["happiness_rank"]
data.head()
#with using set_index means make index happiness_rank and you can not back it as a column
#but if we using data["happiness_rank"] series we can use that feature as index and feature
# Setting index : region is outer country is inner index
data1 = data.set_index(["region","country"]) 
data1
data1.loc[["Western Europe"]] # how to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
# pivoting
df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])
df1
#OR
#df1 = df.set_index(["gender","treatment"])
# lets unstack it
# level determines indexes
df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
data = pd.read_csv("../input/2015.csv")
data.rename(columns={"Economy (GDP per Capita)":"economy","Health (Life Expectancy)":"health","Trust (Government Corruption)":"Trust"}, inplace=True)
data.columns = [each.replace(" ","_") if(len(each.split())>1) else each for each in data.columns]
data.columns = [each.lower() for each in data.columns]
data.head()
data.groupby("region").count()
data.groupby("region").country.count()
data.groupby("region").country.count().sum()
data.groupby("region").country.count().sort_values(ascending=False)
#let find North America counts
data[data["region"]=="North America"].region.count()
data.groupby("region").country.count().sort_values(ascending=False).plot(kind="line")
plt.show()
data.groupby("region").country.count().sort_values(ascending=False).plot(kind="bar")
plt.show()
data.groupby("region").country.count().sort_values(ascending=False).plot(kind="hist",bins=50)
plt.show()
data.groupby("region").country.count().sort_values(ascending=False).plot(kind="box")
plt.show()
data.groupby("region").country.count().sort_values(ascending=False).plot(kind="area")
plt.show()
data.groupby("region").country.count().sort_values(ascending=False).plot(kind="pie")
plt.show()
# according to region take means of other features
data.groupby("region").mean()   # mean is aggregation / reduction method
# there are other methods like sum, std,max or min
# we can only choose one of the feature
data.groupby("region").happiness_score.mean() 
#OR
#df.groupby("region")[["happiness_score"]].mean() 
data.groupby("region").mean().sort_values("happiness_score",ascending=False)
# we can only choose one of the feature
data.groupby("region").happiness_score.max()
# Or we can choose multiple features
data.groupby("region")[["happiness_score","economy"]].mean()
data.groupby("region")[["happiness_score"]].mean() 