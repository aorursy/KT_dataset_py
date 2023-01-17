!pip install yfinance
import pandas as pd
import numpy as np
import datetime as dt 
import yfinance as yf
start = dt.datetime(2015,1,1)
end = dt.datetime(2020,9,10)
ticker = "SPY"

data = yf.download(tickers = ticker, start = start, end = end)
data.head()
data_monthly = data.resample('M').agg(lambda x: x[-1])
data_monthly.head()