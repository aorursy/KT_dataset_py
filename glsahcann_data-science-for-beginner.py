# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#We uses pandas library for add the data.
data = pd.read_csv('../input/DJIA_table.csv')
data.shape
#Lets learn what are the columns names of our data. We call feature 
data.columns
#Learn more about our columns
data.info()
#Lets do Date column's type Datetime  
data.Date= pd.to_datetime(data.Date)  #change of feature type
data.dtypes  #show features type
#Show the top five entries in the database
data.head() 
#Gives the last five entries in the database
data.tail()
#We want to reach High column and 0.index
data.High[0]
#We can do that too 
#data["High"][1]
#data.loc[1,["High"]]
#We can reach High and Low column's line's all
data[["High","Low"]]
#or we can reach the first 5 lines
data.loc[:5,["High","Low"]]
#Gives all columns between Open and Close
data.loc[:3,"Open":"Close"]
filter1 = data.High > 10190
data[filter1]
filter2 = data.Low < 10000
data[filter2]
## Combining filters
data[filter1 & filter2]
#New column is  High and Low column's sum
data["total"] = data.High + data.Low
data.head()
#Lets see how is it.
def operation(x):   #We create a function
    return x/10000
data.Open.apply(operation)  #We use the function with apply() 
#Lets see
data.Open.apply(lambda x: x/10000) #of course we use apply() 
#Learn index name
print(data.index.name)
#If you want to change
data.index.name = "index_name"
print(data.index.name)
data.index = range(1,1990,1)
data.head()
#Create a new column. The column write rise for higher values than average else fall
high_mean = data.High.mean()  #calculate average
data["high_level"] = ["rise" if high_mean < each else "fall" for each in data.High]  #values are scanned in high column
data.head(1000)
#Create a new column. The column write rise for higher values than average else fall
low_mean = data.Low.mean()  #calculate average
data["low_level"] = ["rise" if low_mean < each else "fall" for each in data.Low]  #values are scanned in low column
data.head(1000)
data1 = data.set_index(["high_level","low_level"])
data1.head(1000)
#Delete the total column
data = data.drop(["total"],axis=1)
data.columns
#Values in the high_level column
data.high_level
#Different values in the high_level column
data.high_level.unique()
dictionary = {"Sex":["F","F","M","M"],
              "Size":["S","L","S","L"],
              "Age": [10,48,24,35]}
data_1 = pd.DataFrame(dictionary)
data_1
#We new index sex and  columns size 
data_1.pivot(index = "Sex", columns = "Size", values = "Age")
#We use sex and size column as index
data_2 = data_1.set_index(["Sex","Size"])
data_2
#Choose size for index
data_2.unstack(level=0)
#Choose sex for index
data_2.unstack(level=1)
#I chenge size column and sex column location
data_3 = data_2.swaplevel(0,1)
data_3
#Create list of values of size and age of each sex  column
pd.melt(data_1, id_vars = "Sex" , value_vars = ["Size","Age"])
data_1
#Calculate averange by sex
data_1.groupby("Sex").mean()
#Max value by size
data_1.groupby("Size").max()
#Min value by size
data_1.groupby("Size").min()