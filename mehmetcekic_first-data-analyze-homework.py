# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import datetime
data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
data["Converted_Timestamp"] = [datetime.datetime.fromtimestamp(each) for each in data.Timestamp]

print(data.columns)


data_2012 = data[data.Converted_Timestamp < datetime.datetime(2013,1,1)]
print(data_2012.tail)
# plotting Open, Close, Low and High values of bitcoin in 2012
#
plt.subplot(2,2,1)
plt.plot(data_2012.Converted_Timestamp, data_2012.Open, color="red", label="Open", alpha=0.5)
plt.subplot(2,2,2)
plt.plot(data_2012.Converted_Timestamp, data_2012.Close, color="green", label="Close", alpha=0.5)
plt.subplot(2,2,3)
plt.plot(data_2012.Converted_Timestamp, data_2012.Low, color="blue", label="Low", alpha=0.5)
plt.subplot(2,2,4)
plt.plot(data_2012.Converted_Timestamp, data_2012.High, color="black", label="High", alpha=0.5)
plt.grid()
plt.show()
        
# Analyzing year 2013
data_2013 = data[(data.Converted_Timestamp > datetime.datetime(2013,1,1)) & (data.Converted_Timestamp < datetime.datetime(2014,1,1)) ]
print(data_2013.tail)

plt.subplot(2,2,1)
plt.plot(data_2013.Converted_Timestamp, data_2013.Open, color="red", label="Open", alpha=0.5)
plt.subplot(2,2,2)
plt.plot(data_2013.Converted_Timestamp, data_2013.Close, color="green", label="Close", alpha=0.5)
plt.subplot(2,2,3)
plt.plot(data_2013.Converted_Timestamp, data_2013.Low, color="blue", label="Low", alpha=0.5)
plt.subplot(2,2,4)
plt.plot(data_2013.Converted_Timestamp, data_2013.High, color="black", label="High", alpha=0.5)
plt.grid()
plt.show()
plt.plot(data_2013.Converted_Timestamp,data_2013.Weighted_Price,color="red", label="Open", alpha=0.5)
# Analyzing year 2013
data_2014 = data[(data.Converted_Timestamp >= datetime.datetime(2014,1,1)) & (data.Converted_Timestamp < datetime.datetime(2015,1,1))]
print(data_2014.tail)
plt.subplot(2,2,1)
plt.plot(data_2014.Converted_Timestamp, data_2014.Open, color="red", label="Open", alpha=0.5)
plt.subplot(2,2,2)
plt.plot(data_2014.Converted_Timestamp, data_2014.Close, color="green", label="Close", alpha=0.5)
plt.subplot(2,2,3)
plt.plot(data_2014.Converted_Timestamp, data_2014.Low, color="blue", label="Low", alpha=0.5)
plt.subplot(2,2,4)
plt.plot(data_2014.Converted_Timestamp, data_2014.High, color="black", label="High", alpha=0.5)
plt.grid()
plt.show()
plt.plot(data_2014.Converted_Timestamp,data_2014.Weighted_Price,color="red", label="Open", alpha=0.5)
# General overview 
plt.plot(data.Converted_Timestamp, data.Weighted_Price, color="red",alpha=0.5)
# from graphic at the above, it seems it is nearly stable till 2017
# So let's investigate between 2017 and 2018
data_2017_2018 = data[(data.Converted_Timestamp >= datetime.datetime(2017,1,1)) & (data.Converted_Timestamp <= datetime.datetime(2018,7,1))]
print(data_2017_2018.tail)


plt.plot(data_2017_2018.Converted_Timestamp, data_2017_2018.Weighted_Price, color="red", alpha=0.5)
#Let's look at daily activity between open and close 
plt.plot(data_2017_2018.Converted_Timestamp, data_2017_2018.Open, color="red", alpha=0.5)
plt.plot(data_2017_2018.Converted_Timestamp, data_2017_2018.Close, color="green", alpha=0.5)
# This graphic shows that there is no big difference between daily open and close
#Let's investigate 10,2017 - 01,2018 
data_big_difference = data[(data.Converted_Timestamp >= datetime.datetime(2017,10,1)) & (data.Converted_Timestamp <= datetime.datetime(2018,2,1))]
plt.plot(data_big_difference.Converted_Timestamp, data_big_difference.High, color = "red", alpha=0.5)
plt.plot(data_big_difference.Converted_Timestamp, data_big_difference.Low, color = "green", alpha=0.5)



# Let's investigate top month
data_top_month = data[(data.Converted_Timestamp >= datetime.datetime(2017,12,1)) & (data.Converted_Timestamp <= datetime.datetime(2017,12,31))]
plt.plot(data_top_month.Converted_Timestamp, data_top_month.High, color = "red", alpha=0.5)
plt.plot(data_top_month.Converted_Timestamp, data_top_month.Low, color = "green", alpha=0.5)
# Let's investigate top week
data_top_week = data[(data.Converted_Timestamp >= datetime.datetime(2017,12,14)) & (data.Converted_Timestamp <= datetime.datetime(2017,12,21))]
plt.plot(data_top_week.Converted_Timestamp, data_top_week.High, color = "red", alpha=0.5)
plt.plot(data_top_week.Converted_Timestamp, data_top_week.Low, color = "green", alpha=0.5)
melted_week = pd.melt(frame=data_top_week, id_vars='Converted_Timestamp', value_vars=['Volume_(Currency)', 'Weighted_Price'])
melted_week
# Now learning Pandas Time Series
# After below method we don't need to Converted_Timestamp column
date_list = list(data["Converted_Timestamp"])
datetime_object = pd.to_datetime(date_list)
data["Date"] = datetime_object
data = data.set_index("Date")
data.drop(["Converted_Timestamp"], axis=1, inplace=True)
data
data["2018-06-26":"2018-06-27"]
# Mean values per year
data.resample("A").mean()
#Mean values per month
data.resample("M").mean()
# Filtering data
boolean = data.Weighted_Price > 10000
data[boolean]
#Transforming data using plain python functions
def div(n):
    return n/2
data.High.apply(div)
#Using lambda function
data.High.apply(lambda n : n/2)
#Let's add a new column which show difference between Open and Close
data["Difference"] = data.Close - data.Open
data