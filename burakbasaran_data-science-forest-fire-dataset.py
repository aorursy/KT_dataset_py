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
data = pd.read_csv('../input/forestfires.csv')
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
# Histogram
# bins = number of bar in figure
data.rain.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
series = data['ISI']        # data['Defense'] = series
print(type(series))
data_frame = data[['ISI']]  # data[['Defense']] = data frame
print(type(data_frame))
# 1 - Filtering Pandas data frame
x = data['rain']>1     # There are only 3 pokemons who have higher defense value than 200
data[x]
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['rain']<1) & (data['temp']>30)]
data.head()
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()
 #For example lets look frequency of months types
print(data['month'].value_counts(dropna =False))  # if there are nan values that also be counted

#for example max temp 33.3 min wind 0.4
data.describe() #ignore null entries
# Plotting all data 
data1 = data.loc[:,["FFMC","DMC","DC"]]
data1.plot()
# it is confusing
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
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
# indexing using square brackets
data["temp"][1]
# using loc accessor
data.loc[1,["temp"]]
# Selecting only some columns
data[["temp","rain"]]
# Slicing and indexing series
data.loc[1:10,"temp":"rain"]   # 10 and "Defense" are inclusive
# From something to end
data.loc[1:10,"ISI":] 
# Creating boolean series
boolean = data.temp > 30
data[boolean]
# Combining filters
first_filter = data.temp > 30
second_filter = data.ISI > 10
data[first_filter & second_filter]
# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()
# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["month","temp"]) 
data1.head(100)

