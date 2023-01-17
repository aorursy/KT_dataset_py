# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.info()
data.corr()
# analyze correlation

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
#shows first 5 rows

data.head()
#shows last 5 rows

data.tail()
data.columns
#that's give us numbers of columns and rows

data.shape
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.slope.plot(kind = 'line', color = 'blue',label = 'Slope',linewidth = 1.5,alpha = 1,grid = True,linestyle = ':')

data.oldpeak.plot(color = 'red',label = 'Oldpeak',linewidth = 1.5, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper left') # legend = puts label into plot

plt.xlabel('x axis') # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot') # title = title of plot

plt.show()
# Scatter Plot 

# x = age, y = chol

data.plot(kind='scatter', x='age', y='chol',alpha = 0.5,color = 'red')

plt.xlabel('Age') # label = name of label

plt.ylabel('Chol')

plt.title('Age Chol Scatter Plot') # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.age.plot(kind = 'hist',bins = 50,figsize = (10,10))

plt.show()
# clf() = cleans it up again you can start a fresh

data.age.plot(kind = 'hist',bins = 50)

plt.clf()

# We can't see plot due to clf()
data.describe()
data.boxplot(column="chol",by="age",figsize=(12,12))

plt.show()
#lets create new data for easily understand melt 

data_new = data.head()

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'chol', value_vars= ['age','trestbps'])

melted
# Index is sex

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'chol', columns = 'variable',values = 'value')
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
# axis = 1, lets add dataframes in column

data1 = data['chol'].head()

data2 = data['trestbps'].head()

conc_data_row = pd.concat([data1,data2],axis =1)

conc_data_row
# Lets look at does Heart Diease data have nan value

data.info()
# Lets check sex

data["sex"].value_counts(dropna =False)

# There are 0 NAN value
data1 = data
data1["sex"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true

# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert  data['sex'].notnull().all() 

# returns nothing because we don't have any nan values 

# if it was had, even still returns nothing because we did drop nan values
data["sex"].fillna('empty',inplace = True)
assert  data['sex'].notnull().all() # returns nothing because we do not have nan values
# With assert statement we can check a lot of thing. For example

# assert data.columns[1] == 'Name'

# assert data.Speed.dtypes == np.int
# data frames from dictionary

country = ["England","America"]

population = ["20","22"]

list_label = ["country","population"]

list_col = [country,population]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# Add new columns

df["capital"] = ["london","washington"]

df
# Broadcasting

df["income"] = 0 #Broadcasting entire column

df
# Plotting all data 

data1 = data.loc[:,["trestbps","age","chol"]]

data1.plot()

plt.show()
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="trestbps",y = "age")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "age",bins = 50,range= (0,250),density = True)

plt.show()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "age",bins = 50,range= (0,250),density = True,ax = axes[0])

data1.plot(kind = "hist",y = "age",bins = 50,range= (0,250),density = True,ax = axes[1],cumulative = True)

# plt.savefig('graph.png')

plt.show()
data.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # now data is string

# we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))

# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of heart disease data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

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

# data = pd.read_csv('../input/heart-disease-uci/heart.csv')

# data = data.set_index('age')

# data.head()
# indexing using square brackets

data["chol"][43]
# using column attribute and row label

data.chol[43]
# using loc accessor

data.loc[45,["chol"]]
# Selecting only some columns

data[["trestbps","chol"]]
# Difference between selecting columns: series and dataframes

print(type(data["sex"]))     # series

print(type(data[["sex"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"age":"chol"] 
# Reverse slicing 

data.loc[10:1:-1,"age":"chol"] 
# From something to end

data.loc[1:10,"trestbps":] 
# Creating boolean series

boolean = data.age > 60

data[boolean]
# Combining filters

first_filter = data.age > 60

second_filter = data.chol > 230

data[first_filter & second_filter]
# Filtering column based others

data.age[data.chol<150]
# Plain python functions

def div(n):

    return n/2

data.age.apply(div)
# Or we can use lambda function

data.age.apply(lambda n : n/2)
# Defining column using other columns

data["total_risk"] = data.chol + data.trestbps

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

data3.index = range(100,403,1)

data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("age")

# also you can use 

# data.index = data["age"]
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["age","chol"]) 

data1.head(100)

# data1.loc["43","250"] # how to use indexes
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