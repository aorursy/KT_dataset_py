# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('max_columns', None)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#input the dataset 

#input the data set 

stocks = pd.read_csv("../input/prices.csv", parse_dates=['date'])

stocks.head()
#another useful way to check the description of function is very easy in Python or R 

help(pd.read_csv)
#check thow many methods that could used in a dataframe 

dir(stocks)
#rearrange the order of the dataframe 

stock2 = pd.DataFrame(stocks,columns=['date','open','close','low','high','volume','symbol'])

stock2.head()



#think about this question, how about if we create something(names) not in the dataframe

#pd.DataFrame(stocks,columns=['date','open','close','low','high','volume','symbol','stock_price_average'])
#select single or multiple arrays 

#stocks['symbol']

stock_list = stocks.symbol

stock_list.head()
#select multiple arrays 

#just add one more [] inside a bracket 

stock_multiple = stocks[['date','symbol','open']]

stock_multiple.head()
#select with criteria 

#select all the records with particular ticker and the stock price is over 500 dollars 

google = stocks[(stocks['symbol'] == "GOOG") & ((stocks['open'] >= 300) & (stocks['open'] <= 500))]

google.head()
#drop columns from the dataset 

#drop functions 

drop_example = stocks.drop(columns = ['low'])

drop_example.head()
#Access through index or rows 

#dataframe[begin_index:end_index]

#get top three rows of data 

stocks[:3]

#select between rows 

stocks[5:10]
#select specfic rows 

#stocks[1:10:15]



#it did not work out so what should we do ? 
#select the single row 

t = stocks.iloc[1]

t
#select the last tenth row

a = stocks.iloc[-10]

a
#using iloc in columns 

b = stocks.iloc[:,1]

b.head()
#select multiple columns 

c = stocks.iloc[:,0:2]

c.head()
#how to select specific columns ? 

#d = stocks.iloc[]

d
#get it by labels 

e = stocks.loc[:,'date']

e.head()
#sort by label 

close_price = stocks['close']



type(close_price)

#check out what kinds of functions that could perform on a series 

dir(close_price)
close_price.sort_index(ascending=False)
#sort in dataframe by values 

sorted_stocks = stocks.sort_values(by=['symbol','open'],ascending=True)

sorted_stocks.head()
#aggregate the file groupby symbols 

low = stocks.groupby(['symbol'])['low'].mean()

high = stocks.groupby(['symbol'])['high'].mean()



combined = pd.DataFrame(dict(low_mean = low, high_mean = high)).reset_index()

c = combined.drop('symbol',axis='columns')

c.head()

#calcualte the summary function using apply mapping functions 

#type(combined)

f = lambda x: x.max() - x.min()

c.apply(f)
#or another senario is that you want to know the delta between values and mean value of both columns 

f = lambda x:x - x.mean() 

c.apply(f)

c.head()