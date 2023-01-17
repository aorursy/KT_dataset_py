# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pandas_datareader import data, wb

import datetime

import matplotlib.pyplot as plt

%matplotlib inline
start = datetime.datetime(2006,1,1) #this is the start date

end = datetime.datetime(2016,1,1) # This is latest date of data we will analyze

BAC = data.DataReader('BAC','yahoo', start, end) # here we get stock prices of data by means of yahoo search and make a data frame object

BAC
# we can get the from other banks with the same method as folows:

# CitiGroup

C = data.DataReader("C", 'yahoo', start, end)



# Goldman Sachs

GS = data.DataReader("GS", 'yahoo', start, end)



# JPMorgan Chase

JPM = data.DataReader("JPM", 'yahoo', start, end)



# Morgan Stanley

MS = data.DataReader("MS", 'yahoo', start, end)



# Wells Fargo

WFC = data.DataReader("WFC", 'yahoo', start, end)

MS
#here we create  alis of banks' ticker in order to concatenate them according to these thickers list

tickers=["BAC","C","GS","JPM","MS","WFC"]

bank_stocks=pd.concat([BAC,C,GS,JPM,MS,WFC],axis=1,keys=tickers)

bank_stocks # now we have concatenated all of the banks stock prices as a new data frame name bank_stocks
bank_stocks.columns.names = ['Bank Ticker','Stock Info'] # we create two new column names as Bank Ticker and Stock Info

bank_stocks
# Now our data is ready for explainatory data analysis

for tick in tickers:

    print(tick,bank_stocks[tick]["Close"].max()) #for example this is the max Close price for each bank's stock throughout the time period
# we can get the same outcome by using df.xs() method:

bank_stocks.xs(key="Close",axis=1, level="Stock Info").max()
# here we create a new empty DataFrame called returns in order to see daily price changes of the banks

#pct_change(periods=1, fill_method='pad', limit=None, freq=None, **kwargs)

returns=pd.DataFrame()

for tick in tickers:

    returns[tick + "Return"]=bank_stocks[tick]["Close"].pct_change()

returns.head()
bank_stocks.idxmin() #worst single day returns
bank_stocks.idxmax() #maximum return prices according to dates
bank_stocks["BAC"].max() # maximum prices
bank_stocks["BAC"].min()
import seaborn as sns 

sns.pairplot(returns)# here we create a pairplot using seaborn of the returns dataframe to see relations between banks returns of different banks
returns.idxmin() #returns each bank stock had the worst return
returns.idxmax()#returns each bank stock had the best returns
# df.std() method is very useful because it returns the standart deviations of each bank:

# if standart deviation is larger than others, it means that it is the riskiest among the other

returns.std()

# here we understand that Citibank is the riskiest bank 
returns.loc["2015-01-01":"2015-12-30"].std() # this returns the standart deviation of banks in 2015
plt.figure(figsize=(15,10))

sns.distplot(returns["MSReturn"].loc["2015-01-01":"2015-12-30"],bins=50)

# here we create a distplot using seaborn of the 2015 returns for Morgan Stanley
plt.figure(figsize=(15,10))

sns.distplot(returns["CReturn"].loc["2008-01-01":"2008-12-30"],bins=50,color="red")

# here we create a distplot using seaborn of the 2008 returns for CitiGroup 
# better visualization quality I will import plotly library

import plotly

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True) #this will connect our notebook to the library

cf.go_offline()

close=bank_stocks.xs(key="Close",axis=1, level="Stock Info")

close.iplot()

# here we create a line plot showing Close price for each bank for the entire index of time.
bank_stocks["BAC"]["Close"].iplot()

# we can get this line plot for each bank separately as seen below
bank_stocks.xs(key="Volume",axis=1, level="Stock Info").iplot() 

# here we plot the volume of stocks per bank
bank_stocks.xs(key="Close",axis=1, level="Stock Info").mean().iplot(title='The Overall Average Close Prices of Banks in 2008', xTitle='Banks')

bank_stocks["BAC"]["Close"].loc["2008-01-01":"2009-01-01"].rolling(window=30).mean().iplot(title='The Monthly Average of Close Prices Bank of America in 2008', xTitle='Months')

#here we plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008
#we can show the two plot with in a single plot figure in order to make better comparisons:

plt.figure(figsize=(20,10))

bank_stocks["BAC"]["Close"].loc["2008-01-01":"2009-01-01"].plot(label="BAC Close",title='The Overall Average Close Prices of Bank of America in 2008 vs The Monthly Average Close Prices of Bank of America in 2008')

bank_stocks["BAC"]["Close"].loc["2008-01-01":"2009-01-01"].rolling(window=30).mean().plot(label="BAC Monthly Average")

plt.legend()

# so we can see general trend of months versus daily changes
bank_stocks.corr()

# here we get the correlation values between banks
bank_stocks.xs(key="Close",axis=1, level="Stock Info")#here is closing prices of different banks
bank_stocks.xs(key="Close",axis=1, level="Stock Info").corr() 

# here we can see the correlation between closing prices of different banks
plt.figure(figsize=(20,10))

sns.heatmap(bank_stocks.xs(key="Close",axis=1, level="Stock Info").corr(),cmap="coolwarm",linecolor="white", linewidths=1,annot=True)

#This is the heatmap of the correlation between close prices of different banks
sns.clustermap(bank_stocks.xs(key="Close",axis=1, level="Stock Info").corr(),cmap="coolwarm",linecolor="white", linewidths=1,annot=True)

#This is the clustertmap of the correlation between close prices of different banks that provides relational heatmap
# here we use .iplot(kind='candle) to create a candle plot of Bank of America's stock from Jan 1st 2015 to Jan 1st 2016

bank_stocks["BAC"].loc["2015-01-15":"2016-01-03"].iplot(kind="candle")

# we can see interactive detailed stock prices of each day like opening and closing price
bank_stocks["MS"]["Close"].loc["2015-01-01":"2015-12-31"].rolling(window=30).mean().iplot()
bank_stocks["MS"]["Close"].loc["2015-01-01":"2015-12-31"].ta_plot(study="sma",periods=[7,30,55])

# SMA means  simple moving average closing prices in the concerned period of time

#here we can see 7, 30 and 55 days periods of Close price changes of Morgan Stanley in 2015 in the plotly iplot
bank_stocks["BAC"]["Close"].loc["2015-01-01":"2015-12-31"].ta_plot(study="boll") 

#here we create a Bollinger Band Plot for Bank of America for the year 2015.
