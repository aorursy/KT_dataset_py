import pandas_datareader.data as web

import datetime

import pandas as pd

import numpy as np



start = datetime.datetime(2018,1,1)

end = datetime.date.today()

 

apple = web.DataReader("AAPL", "yahoo", start, end)

print(apple.columns)