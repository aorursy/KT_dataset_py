# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
data.info()
data.columns
data.head(10)
import datetime

data['date'] = pd.to_datetime(data['Timestamp'],unit='s') # timestamp convert to datetime
data["date_new"] = data['date'].apply(lambda x: x.strftime('%Y')) # datetime convert to year ( day, month)
data['date_new2']=data['date_new'].str.replace('-','').apply(int) # type of datetime convert type of integer 
data.drop(["date"],axis=1,inplace = True) 
data.drop(["date_new"],axis=1,inplace = True)
data = data.rename(columns={ 'Volume_(Currency)':'Volume_Currency', 'Volume_(BTC)': 'Volume_BTC', 'date_new2':'year'})

print(data.head(5))
print(data.info())
# counting column
# data["year"].count
# how many different values in column
data["year"].nunique()
print(data['Weighted_Price'].min())
print(data['Weighted_Price'].max())
data.corr()
data.columns
plt.scatter(data.year, data.Volume_Currency, color = "g", alpha=0.5)
plt.xlabel("year")
plt.ylabel("Volume_Currency")
plt.title("Volume_BTC/Currency")
plt.show()
plt.scatter(data.year, data.Weighted_Price, color = "r", alpha=0.5)
plt.xlabel("year")
plt.ylabel("Weighted_Price")
plt.title("Weighted_Price/Currency")
plt.show()


#data[(data["Open"]<data["Close"]) & (data['year']==2017)]
year_2011= data[data.year == 2011]
year_2012= data[data.year == 2012]
year_2013= data[data.year == 2013]
year_2014= data[data.year == 2014]
year_2015= data[data.year == 2015]
year_2016= data[data.year == 2016]
year_2017= data[data.year == 2017]
year_2018= data[data.year == 2018]
print(year_2011.Weighted_Price.max())
print(year_2012.Weighted_Price.max())
print(year_2013.Weighted_Price.max())
print(year_2014.Weighted_Price.max())
print(year_2015.Weighted_Price.max())
print(year_2016.Weighted_Price.max())
print(year_2017.Weighted_Price.max())
print(year_2018.Weighted_Price.max())
#plt.plot(asd.date_new2, asd.Weighted_Price, color = "red", label= "asd")
plt.hist(year_2011.Weighted_Price,bins=5,color="pink") 
plt.hist(year_2012.Weighted_Price,bins=5,color="orange")
plt.hist(year_2013.Weighted_Price,bins=5,color="grey") 
plt.hist(year_2014.Weighted_Price,bins=5,color="y") 
plt.hist(year_2016.Weighted_Price,bins=5,color="r") 
plt.hist(year_2017.Weighted_Price,bins=5,color="g") 
plt.hist(year_2018.Weighted_Price,bins=5,color="b") 
plt.show()
