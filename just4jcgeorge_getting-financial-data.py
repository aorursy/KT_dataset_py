!pip install quandl

import os

import pandas as pd

import pandas_datareader.data as web

from pandas_datareader import data

from datetime import datetime

from pprint import pprint
import quandl

import datetime



start = datetime.datetime(1980,1,1)

end = datetime.datetime(2019,12,23)

 

AAPL = quandl.get ("WIKI/AAPL", start_date=start, end_date=end) 
AAPL.info()

AAPL.head()
# First day

start_date = '2014-01-01'

# Last day

end_date = '2018-01-01'

# Call the function DataReader from the class data

goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
print(goog_data)
sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

sp500_constituents = pd.read_html(sp_url, header=0)[0]
sp500_constituents.info()
sp500_constituents.head()
start = '2014'

end = datetime.datetime(2017, 5, 24)



yahoo= web.DataReader('FB', 'yahoo', start=start, end=end)

yahoo.info()
# Create you own IEX API KEY at https://iexcloud.io/

os.environ["IEX_API_KEY"] = '' # <- Enter your IEX_API_KEY here

start = datetime.datetime(2015, 2, 9)

# end = datetime(2017, 5, 24)



iex = web.DataReader('FB', 'iex', start)

iex.info()
iex.tail()
book = web.get_iex_book('FB')
list(book.keys())
orders = pd.concat([pd.DataFrame(book[side]).assign(side=side) for side in ['bids', 'asks']])

orders.head()
for key in book.keys():

    try:

        print(f'\n{key}')

        print(pd.DataFrame(book[key]))

    except:

        print(book[key])
symbol = 'FB.US'





AAPL = quandl.get ("WIKI/FB", start_date=start, end_date=end) 

AAPL.info()

AAPL