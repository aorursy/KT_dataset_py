# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Country.csv') # Show the path you will use
data.info() # Display content of data 
data.head() # Display first 5 rows of data
#data.tail
data.columns # Display first all columns of data
# data frames from dictionary
country = ["Spain","Turkey"]
print(country)
population = ["100","200"]
print(population)
list_label = ["countries","populations"]
print(list_label)
print('')
list_col = [country,population]
print(list_col)
zipped = list(zip(list_label,list_col))
print(zipped)
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df

# Adding new columns
df["capital"]=["Madrid","Ankara"]
df

# Broadcasting
df["income"] = 0 # Broadcasting entire column
df
data.head()
# Plotting all data 
data1 = data.loc[:,["LatestIndustrialData","LatestTradeData","LatestWaterWithdrawalData"]]
data1.plot()
# graph is confusing, lets create subplots
# subplots let us seperate graphs
data1.plot(subplots = True)
plt.show()
# scatter plot usage
data1.plot(kind = "scatter",x="LatestTradeData",y = "LatestWaterWithdrawalData")
plt.show()
# histogram plot 
data1.plot(kind="hist",y = "LatestWaterWithdrawalData",bins = 10)
# histogram subplot with non-cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "LatestWaterWithdrawalData",bins = 50,normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "LatestWaterWithdrawalData",bins = 50,normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
# Lets take a look at statistical information
data.describe()
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however, we want it to be a datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# close warning (not about this part)
import warnings
warnings.filterwarnings("ignore")

# In order to practice, lets take a look at head of world development data and add it to a time list
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
# As you can see there are a lot of nan values because data2 does not include all months
# In real life (Data is real. Not created from us like data2), we can solve this problem with interpolate
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")
# Take a look at last columns
# Or we can interpolate with mean()
data2.resample("M").mean().interpolate("linear")
#Changing index of our data
#data = data.set_index("CountryCode")   
#data.head()
# indexing using square brackets (same as below)
data["Alpha2Code"][1]

# using column attribute and row label
# data.Alpha2Code[1]
# Selecting only some columns
data[["ShortName","LongName"]]
# Difference between selecting columns: series and dataframes
print(type(data["ShortName"]))     # series
print(type(data[["ShortName"]]))   # data frames
# Slicing and indexing series
data.loc[1:10,"Alpha2Code":"Region"]   # 10 and "Defense" are inclusive
# Reverse slicing 
data.loc[10:1:-1,"Alpha2Code":"Region"]
# From something to end
data.loc[1:10,"LatestAgriculturalCensus":]
# Creating boolean series
boolean = data.LatestWaterWithdrawalData > 2011
data[boolean]
# Combining filters
first_filter = data.LatestIndustrialData > 2008
second_filter = data.LatestWaterWithdrawalData > 2011
data[first_filter & second_filter]
# Filtering column based others
data.LatestIndustrialData[data.LatestIndustrialData<2008]
# Applying python functions to data
def div(n):
    return n/2
data.LatestIndustrialData.apply(div)
# Or we can use lambda function for shorter way
data.LatestIndustrialData.apply(lambda n : n/2)
# Defining column using other columns
data["superiors"] = data.LatestIndustrialData + data.LatestTradeData  #just an example
data.head()
# this shows our index name:
print(data.index.name)   #None
# lets change it
data.index.name = "index_name"
data.head()
# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just an example
data3.index = range(100,347,1)
data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# It was like this;
# data = data.set_index("CountryCode")
# also you can use; 
# data.index = data["CountryCode"]
dic = {"sports":["Football","Football","Basketball","Basketball"],"gender":["F","M","F","M"],"skill":[10,45,32,9],"age":[15,20,52,43]}
df = pd.DataFrame(dic)
df
# pivoting
df.pivot(index="sports",columns = "gender",values="skill")
df1 = df.set_index(["sports","gender"])
df1
# lets unstack it
# level determines indexes
df1.unstack(level=0)

# or
#df1.unstack(level=1)
# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2
# df.pivot(index="sports",columns = "gender",values="response")
pd.melt(df,id_vars="sports",value_vars=["age","skill","gender"])
# according to sports take means of other features
df.groupby("sports").mean()   # mean is aggregation / reduction method
# there are other methods like sum, std, max or min
# we can choose specific feature
df.groupby("sports").age.max()
# Or we can choose multiple features
df.groupby("sports")[["age","skill"]].min() 