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

import pandas as pd

import numpy as np

import datetime

%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# Optional Plotly Method Imports

import plotly

import cufflinks as cf

cf.go_offline()
tickers = ["BAC","C","GS","JPM","MS","WFC"] #bank stock tickers list
start_date = "01-01-2006"

end_date = "01-01-2016" #dates of exploration
BAC = data.DataReader("BAC","yahoo",start_date,end_date)

C =  data.DataReader("C","yahoo",start_date,end_date)

GS =  data.DataReader("GS","yahoo",start_date,end_date)

JPM =  data.DataReader("JPM","yahoo",start_date,end_date)

MS =  data.DataReader("MS","yahoo",start_date,end_date)

WFC =  data.DataReader("WFC","yahoo",start_date,end_date)

    

#using data reader to create dataframes with prices and dates for each ticker 

WFC.head()
# using pd concat to make one dataframe and setting keys argument to equal tickers list

bank_stocks = pd.concat([BAC,C,GS,JPM,MS,WFC],axis = 1,keys = tickers)





bank_stocks.head()
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
#creating a line plot for the entire index time using cufflinks and plotly

bank_stocks.xs(key = "Close",axis = 1,level = "Stock Info").iplot()
# creating a candle chart for BAC for the entire index period using cufflinks

bac15 = BAC[["Open","High","Low","Close"]].loc["2008-01-01":"2016-01-01"]

bac15.iplot(kind = "candle")
# Bollinger bands for the year 2015

BAC["Close"].loc["2015-01-01":"2016-01-01"].ta_plot(study = "boll")