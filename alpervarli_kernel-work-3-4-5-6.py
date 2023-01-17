# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns # for visulisation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Read as DataFrame

data=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1') 

data.head()  # head shows first 5 rows
# tail shows last 5 rows

data.tail()
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble

data.shape
data.info() # There are 181691 terrorist attack record for All Countries
data.corr()
#EXPLORATORY DATA ANALYSIS (EDA)

# For example lets look frequency of country_txt

print(data['country_txt'].value_counts(dropna =False))  # if there are nan values that also be counted

# As it can be seen below there are 205 country

#There is max attack in Iraq





#There is max attack in 2014 year

print(data['iyear'].value_counts(dropna =False))
# For example max HP is 255 or min defense is 5

data.describe() #ignore null entries
#compare iyear and attacktype1

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers 



data.boxplot(column='iyear',by = 'attacktype1')
#tidy data with melt().

data_new = data.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'country_txt', value_vars= ['city','iyear'])

melted
# PIVOTING DATA

#Reverse of melting.

# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'country_txt', columns = 'variable',values='value')



# CONCATENATING DATA

#We can concatenate two dataframe 

# Firstly lets create 2 data frame

# axis = 0 : adds dataframes in row

data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row

# axis = 0 : adds dataframes in columns

data1 = data['country'].head(181691)

data2 = data['country_txt'].head(181691)

data3= data['iyear'].head(181691)

conc_data_col = pd.concat([data1,data2,data3],axis =1) 

conc_data_col

data.dtypes
# MISSING DATA and TESTING WITH ASSERT

# As you can see there are 181691  entries.There is no non-null record

data.info()

#4. PANDAS FOUNDATION

# data frames from dictionary

country = [data.country_txt]

year = [data.iyear]

list_label = ["country","year"]

list_col = [country,year]

zipped = list(zip(list_label,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df





# Add new columns  # data frame e yeni columns ekle ve değer ata

df["Attack_Type_1"] = [data.attacktype1_txt]

df
#VISUAL EXPLORATORY DATA ANALYSIS

# Plotting all data 

data1 = data.loc[:,["targtype1","attacktype1","country"]]

data1.plot()





data1.plot(subplots = True)

plt.show()
# scatter plot   



data1.plot(kind = "scatter",x="country",y = "attacktype1")

plt.show()
# hist plot  

# histogram bize frekans ölçüyor 

# range y ekseni (0-250)

# normed datanın normalize edilmiş hali 0-1 arasında (x ekseni)eğer bunu yazmazsak sayıları görürüz

data1.plot(kind = "hist",y = "country",bins = 5)

#,range= (0,250),normed = True
# histogram subplot with non cumulative and cumulative



fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "country",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "country",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
# STATISTICAL EXPLORATORY DATA ANALYSIS



data.describe()
# INDEXING PANDAS TIME SERIES

data2 = data.head(181681)

data2['time_list2'] = data["iyear"].map(str) +'-'+ data["imonth"].map(str)+'-'+ data["iday"].map(str)

first_filter = data2.imonth > 0 

second_filter= data2.iday>0

data3=data2[first_filter & second_filter]

print (data3['time_list2'])

print(type(data3.time_list2))









data3

                           





#str([data3.time_list2])



print(type(data3.time_list2))

#datetime_object = pd.to_datetime(data3.time_list2)

#print(type(datetime_object))
# There is allready date fields in data

#import warnings

#warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

#data2 = data.head()

#date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(data3['time_list2']) # date time dönüştürdük

data2["date"] = datetime_object #date field olarak dataya ekledik

# lets make date as index

data2= data2.set_index("date")  # data mı tarih indexli yaptım. 

#ime series data halin geldi

data2 



# Now we can select according to our date index

#print(data2.loc["1970-01-02"])

print(data2.loc["1970-01-02":"1970-01-19"])
# RESAMPLING PANDAS TIME SERIES

# We will use data2 that we create at previous part

data2.resample("A").mean()

# Lets resample with month

data2.resample("M").mean()
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
data.info
# MANIPULATING DATA FRAMES WITH PANDAS

# INDEXING DATA FRAMES



# read data 

#data=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1') 

#data= data.set_index("id")

#data.head()



# indexing using square brackets : 

data["iyear"][1]
# using column attribute and row label  .same result

data.iyear[1]
# using loc accessor 

data.loc[1,["iyear"]]
data[["iyear","country_txt"]]
# SLICING DATA FRAME

# Difference between selecting columns: series and dataframes 



print(type(data["country_txt"]))     # series 

print(type(data[["country_txt"]]))  # data frames

# Slicing and indexing series

data.loc[1:10,"iyear":"country_txt"]   # 10 and "country_txt" are inclusive 
# Reverse slicing

data.loc[1:10:-1,"iyear":"country_txt"]

# From something to end : 

data.loc[1:10,"iyear":] 
# FILTERING DATA FRAMES very important lesson

# Creating boolean series: 

boolean = data.iyear > 2010

data[boolean]

# Combining filters

first_filter = data.iyear > 2010

second_filter = data.country > 130

data[first_filter & second_filter]
# Filtering column based others: 

data.country_txt[data.country<130]
# TRANSFORMING DATA

# Plain python functions

def div(n):

    return n/2

data.attacktype1.apply(div)





# Or we can use lambda function: same result 

data.attacktype1.apply(lambda n : n/2)
# INDEX OBJECTS AND LABELED DATA

# our index name is this:

print(data.index.name) # if there is  we are looking index column

# lets change it

data.index.name = "index_name" #if there is  we are changing index name

data.head()

# Overwrite index

# if we want to modify index we need to change all of them.

#data.head()

# first copy of our data to data3 then change index 

#data4 = data.copy() 

# lets make index start from 100. It is not remarkable change but it is just example

#data4.index = range(100,900,1) # 100 den başla 1 er artarak index tanımla

#data4.head()
# HIERARCHICAL INDEXING

# Setting index : type 1 is outer type 2 is inner index

data5 = data.set_index(["country_txt","iyear"])  # country_txt: 1. index column (outer) , iyear : 2. index column(inner)

data5.head(100)
data4 = data.copy()
# MELTING DATA FRAMES

pd.melt(data4,id_vars="country_txt",value_vars=["city","iyear"])

# CATEGORICALS AND GROUPBY

# according to country_txt take means of other features

data4.groupby("country_txt").mean()

data4.groupby("country_txt").weaptype1.max() 
# Or we can choose multiple features

data4.groupby("country_txt")[["iyear","attacktype1_txt"]].max()
data4.info()
#Attack count according to the countries

groupped =data4.groupby(['country_txt']).iyear.agg('count').to_frame('Attack_Count_Year').reset_index()



groupped

# Attack Count - Country Code

groupped.plot(subplots = True)

plt.show()
#Attack count according to the city and years

groupped_city=data.groupby(['country_txt','iyear','city']).iyear.agg('count').to_frame('Attack_Count_Year').reset_index()

groupped_city
#Attack count according to the years

groupped2=data.groupby(['country_txt','iyear']).iyear.agg('count').to_frame('Attack_Count_Year').reset_index()

groupped2

#Attack count according to the years

groupped2.plot(kind = "scatter",x="iyear",y = "Attack_Count_Year")

plt.show()
groupped2.boxplot(column='Attack_Count_Year',by = 'iyear')