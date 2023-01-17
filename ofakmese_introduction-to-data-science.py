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
data = pd.read_csv('../input/master.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.year.plot(kind = 'line', color = 'r',label = 'year',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.')

data.suicides_no.plot(color = 'g',label = 'suicides_no',linewidth=1, alpha = 0.5,grid = True,linestyle = ':')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Line Plot 

# x = year, y = suicides_no

data.plot(kind='line', x='year', y='suicides_no',alpha = 0.5,color = 'blue', grid = True,linestyle = ':', figsize = (10,8))

plt.xlabel('year')              # label = name of label

plt.ylabel('suicides_no')

plt.title('year, suicides_no Scatter Plot')            # title = title of plot
# Scatter Plot 

# x = year, y = suicides_no

data.plot(kind='scatter', x='year', y='suicides_no',alpha = 0.5,color = 'red', figsize = (8,6))

plt.xlabel('year')              # label = name of label

plt.ylabel('suicides_no')

plt.title('year suicides_no Scatter Plot')            # title = title of plot
plt.scatter(data.year, data.suicides_no, color = "red", alpha = 0.5) # other notation

plt.xlabel('year')              # label = name of label

plt.ylabel('suicides_no')

plt.title('year suicides_no Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.year.plot(kind = 'hist',bins = 40,figsize = (6,6))

plt.show()
# clf() = cleans it up again you can start a fresh

data.year.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
data = pd.read_csv('../input/master.csv') # data import 
series = data['year']        # data['Defense'] = series

print(type(series))

data_frame = data[['suicides_no']]  # data[['Defense']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['year']>2014     # There are only 3 pokemons who have higher defense value than 200

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['year']>2014, data['suicides_no']>5000 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['year']>2014) & (data['suicides_no']>5000)] # other notation
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



# For pandas we can achieve index and value 

# select specific indexes in a column

for index,value in data[['year']][47:49].iterrows(): # 47 include, 49 exclude

    print(index," : ",value)
# lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.suicides_no)/len(data.suicides_no)

print("threshold: ", threshold)

data["suicides_no_level"] = ["High" if i > threshold else "Low" for i in data.suicides_no]  # list comprehension

data.loc[1450:1460,["suicides_no_level","suicides_no"]] # we will learn loc more detailed later
data = pd.read_csv('../input/master.csv')

data.head()  # head shows first 5 rows
# tail shows last 5 rows

data.tail()
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
# For example lets look frequency of sex types

print(data['sex'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 13910 male, 13910 female  
data.describe() #ignore null entries
# For example: comparison of generation by year

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

# outlier data is not seen in this analysis

data.boxplot(column='year',by = 'generation', figsize = (10,8)) 
# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'country', value_vars= ['year','suicides_no'])

melted
# Firstly lets create 2 data frame

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['country'].head()

data2= data['year'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in row

conc_data_col
data.dtypes
# lets convert object(str) to categorical

data['sex'] = data['sex'].astype('category')
data.dtypes
# Lets look at does master data have nan value

data.info()
# Lets chech HDI for year

data["HDI for year"].value_counts(dropna =False)

# As you can see, there are 19456 NAN value
# Lets drop NaN values

data1=data   # also we will use data to fill missing value so I assign it to data1 variable

data1["HDI for year"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
assert  data['HDI for year'].notnull().all() # returns nothing because we drop NaN values
data["HDI for year"].fillna('empty',inplace = True)
assert  data['HDI for year'].notnull().all() # returns nothing because we do not have nan values
country = ["Spain","France"]

population = ["11","12"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["capital"] = ["madrid","paris"]

df
df["income"] = 0 #Broadcasting entire column

df
# Plotting all data 

data1 = data.loc[:,["suicides_no","population","gdp_per_capita ($)"]]

data1.plot()

# it is confusing
data1.plot(subplots = True, figsize=(10,8))

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="suicides_no",y = "gdp_per_capita ($)", figsize=(10,8))

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "suicides_no",bins = 50,range= (0,250),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "suicides_no",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "suicides_no",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
data.head() # index of our data is 012345
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date") # index of our data is date

data2 
# Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
# We will use data2 that we create at previous part

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
# read data

data = pd.read_csv('../input/master.csv')

data.head()
# indexing using square brackets

data["generation"][0]
# using column attribute and row label

data.generation[0] # a different method
# using loc accessor

data.loc[0,["generation"]]
# Selecting only some columns

data[["age","generation"]]
# Difference between selecting columns: series and dataframes

print(type(data["generation"]))     # series

print(type(data[["generation"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"age":"population"]   # 10 and "Defense" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"age":"population"] 
# From something to end

data.loc[0:10,"country-year":] 
boolean = data.year > 2000

data[boolean]
# Combining filters

first_filter = data.year > 2014

second_filter = data.suicides_no > 5000

data[first_filter & second_filter]
# Filtering column based others

data.year[data.suicides_no>20000]
# Plain python functions

def div(n):

    return n/2

data.suicides_no.apply(div)
# Or we can use lambda function

data.suicides_no.apply(lambda n : n/2)
# Defining column using other columns

data["total_power"] = data.population + data.suicides_no

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# first copy of our data to data3 then change index 

data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(100,27920,1)

data3.head()
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/master.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["year","generation"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # howw to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

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
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max() 
df.groupby("treatment")[["age","response"]].min() 