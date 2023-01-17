# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')

data.head(10)

#shows last 5 rows

data.tail()

#give number of rows and columns in a tuble

data.shape
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')              # label = name of label

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.Speed.plot(kind = 'hist',bins = 50,figsize = (10,10))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Speed.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()

series = data['Defense']        # data['Defense'] = series

print(type(series))

data_frame = data[['Defense']]  # data[['Defense']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200

data[x]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['Defense']>200) & (data['Attack']>100)]
#List comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later
#to call Type 1

#we can use data['Type 1']
#value_counts(): Frequency counts

    
# For example lets look frequency of pokemom types

print(data['Type 1'].value_counts(dropna = False))  # if there are nan values that also be counted

# As it can be seen below there are 112 water pokemon or 70 grass pokemonprint
#outliers: the value that is considerably higher or lower from rest of the data
# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Green line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# Circles are outliers

data.boxplot(column='Attack',by = 'Legendary')

plt.show()
# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head()    # I only take 5 rows into new data

data_new

# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['HP','Attack','Defense'])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Name', columns = 'variable',values='value')

# Firstly let's create 2 data frame in a vertical way

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
# Now let's create 2 data frame in a horizontal way

data1 = data['Attack'].head()

data2= data['Defense'].head()

data3=data['Name'].head()

conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 1 : adds dataframes in column

conc_data_col
data.dtypes
# lets convert object(str) to categorical and int to float.

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')
# As you can see Type 1 is converted from object to categorical

# And Speed ,s converted from int to float

data.dtypes
data.head()

# Lets look at does pokemon data have nan value

# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.

data.info()
# Lets chech Type 2

data["Type 2"].value_counts(dropna =False)

# As you can see, there are 386 NAN value
# Lets drop nan values

data1=data   # also we will use data to fill missing value so I assign it to data1 variable

data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
data["Type 2"].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

# assert data.columns[1] == 'Name'

# assert data.Speed.dtypes == np.int
# Plotting all data 

data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()

# it is confusing

plt.show()
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()
#hist plot 

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),density = True)

plt.show()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),density = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),density = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt

plt.show()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)  #it becomes datetime index

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object



data2
# lets make date as index

data2= data2.set_index("date")

data2 
# Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-15"])
# We will use data2 that we create at previous part

#resample to the years and take the average of them

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.HP.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
data.head()
data = data.set_index("#")

data.head()
# indexing using square brackets

data["HP"][1]

#otherwise it shoould be data["HP"][0]
# using column attribute and row label

data.HP[1]
# using loc accessor

data.loc[1,["HP"]]
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

data.loc[1:10,"Speed":] 
# Creating boolean series

boolean = data.HP > 200

data[boolean]
# Combining filters

first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]
# Filtering column based others

#hıza göre filtrelediğim pokemonların canını göster

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

data3.index = range(100,900,1)

data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("#")

# also you can use 

# data.index = data["#"]

# lets read data frame one more time to start from beginning

data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')

data.head(10)



# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Type 1","Type 2"]) 

data1.head(20)

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

pd.melt(df,id_vars="treatment",value_vars=["age","response"]) #hiçbir şey yazmazsak default olarak variable ve value ekliyor
# We will use df

df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max() 
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["gender"] = df["gender"].astype("category")

#df["treatment"] = df["treatment"].astype("category")

#df.info()