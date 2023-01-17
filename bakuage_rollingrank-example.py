!pip install rollingrank
import numpy as np

import pandas as pd
import pandas_datareader.data as web

df = web.DataReader("usmv", "yahoo", "1980/1/1").dropna()

display(df)
df['Adj Close'].plot()
import rollingrank

df['adj_close_rank'] = rollingrank.rollingrank(df['Adj Close'], window=100)

df['adj_close_rank'].plot()