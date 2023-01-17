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
data=pd.read_csv("../input/avocado.csv")
data.info()
data.corr()
#correlation map
f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".2f",ax=ax)
plt.show()
data.head(10)
data.rename( columns={'Unnamed: 0':'new column name'}, inplace=True )
data = data.drop('new column name', axis=1)
data.columns
data.AveragePrice.plot(kind='line',color='b',label='Average Price',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
data.plot(kind='scatter', x='Total Volume', y='AveragePrice',alpha = 0.5,color = 'red') #column namaler birebir aynı olmalı
plt.xlabel('Total Volume')              # label = name of label
plt.ylabel('Average Price')
plt.title('Scatter Plot')            # title = title of plot
data.rename( columns={'Small Bags':'SmallBags'}, inplace=True )
data.SmallBags.plot(kind = 'hist',bins = 10,figsize = (12,12)) #figsize framein boyunu değiştiriyor
plt.show()
# 1 - Filtering Pandas data frame
x = data['AveragePrice']<1     # There are only 3 pokemons who have higher defense value than 200
data[x]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['AveragePrice']<1) & (data['region']=='West') & (data['type']=='organic')]
x = 7
def f():
    y = 2*x        # there is no local scope x
    return y
print(f())         # it uses global scope x
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
f(country = 'trabzon', capital = 'sürmene', population = 45455)
# zip example
name = ["ayse","berna","sefa","seyma"]
age = [15,6,7,8]
z = zip(name,age)
print(z)
z_list = list(z)
print(z_list)
# lets return  csv and make one more list comprehension example
data=pd.read_csv("../input/avocado.csv")
threshold = sum(data.AveragePrice)/len(data.AveragePrice)
data["price_level"] = ["high price" if i > threshold else "low price" for i in data.AveragePrice]
data.loc[:30,["price_level","AveragePrice"]] # we will learn loc more detailed later
data = pd.read_csv('../input/avocado.csv')
data.head()  # head shows first 5 rows
# data frames from dictionary
city = ["Ankara","Trabzon"]
altitude = ["1000","1"]
list_label = ["city","altitude"]
list_col = [city,altitude]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df["region"] = ["Ic Anadolu","Karadeniz"]
df
df["population"] = 0 #Broadcasting entire column
df
# Plotting all data 
data1 = data.loc[:,["4046","4225","4770"]]
data1.plot()

# subplots
data1.plot(subplots = True)
plt.show()
# scatter plot  
data.plot(kind = "scatter",x="Total Volume",y = "Total Bags")
plt.show()
data.plot(kind = "hist",y = "AveragePrice",bins = 50,range= (0,4),normed = True)
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
data = pd.read_csv('../input/avocado.csv')
data= data.set_index("Date")
data.head()
# Now we can select according to our date index
print(data.loc["2015-12-27"])
print(data.loc["2015-12-27":"2015-12-06"])
# indexing using square brackets
data["AveragePrice"][1]
# using column attribute and row label
data.AveragePrice[1]
# Selecting only some columns
data[["AveragePrice","Total Bags"]]
# Slicing and indexing series
data=pd.read_csv("../input/avocado.csv")
data.loc[1:10,"AveragePrice":"Total Bags"]   # 10 and "Defense" are inclusive
# Creating boolean series
boolean = data.AveragePrice > 1.5
data[boolean]
# Combining filters
first_filter = data.AveragePrice > 1.5
second_filter = data.region =="Albany" 
data[first_filter & second_filter]
# Plain python functions
def div(n):
    return n*2
data.AveragePrice.apply(div)
# Or we can use lambda function
data.AveragePrice.apply(lambda n : n*2)
data1 = data.set_index(["AveragePrice","region"]) 
data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
df.pivot(index="treatment",columns = "gender",values="response")

