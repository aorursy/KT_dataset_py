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
!pip install yahoo-finance

!pip install yfinance --upgrade --no-cache-dir

!pip install fix-yahoo-finance
import pandas as pd

table123=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

df123 = table123[0]

df123.to_csv('S&P500-Info.csv')

df123.to_csv("S&P500-Symbols.csv", columns=['Symbol'])

import yfinance as yf

from yahoo_finance import Share

import bs4 as bs

import datetime as dt

import os

from pandas_datareader import data as pdr

import pickle

import requests

import fix_yahoo_finance as yf



yf.pdr_override



def save_sp500_tickers():

    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    soup = bs.BeautifulSoup(resp.text, 'lxml')

    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []

    for row in table.findAll('tr')[1:]:

        ticker = row.findAll('td')[0].text.replace('.', '-')

        ticker = ticker[:-1]

        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:

        pickle.dump(tickers, f)

    return tickers



save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False):

    if reload_sp500:

        tickers = save_sp500_tickers()

    else:

        with open("sp500tickers.pickle", "rb") as f:

            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):

        os.makedirs('stock_dfs')

    start = dt.datetime(2000, 6, 8)

    end = dt.datetime.now()

    for ticker in tickers:

        print(ticker)

        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):

            df = pdr.get_data_yahoo(ticker, start, end)

            df.reset_index(inplace=True)

            df.set_index("Date", inplace=True)

            df.to_csv('stock_dfs/{}.csv'.format(ticker))

        else:

            print('Already have {}'.format(ticker))

get_data_from_yahoo()



def compile_data():

    with open("sp500tickers.pickle", "rb") as f:

        tickers = pickle.load(f)



    main_df = pd.DataFrame()



    for count, ticker in enumerate(tickers):

        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))

        df.set_index('Date', inplace=True)



        df.rename(columns={'Adj Close': ticker}, inplace=True)

        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)



        if main_df.empty:

            main_df = df

        else:

            main_df = main_df.join(df, how='outer')



        if count % 10 == 0:

            print(count)

    print(main_df.head())

    main_df.to_csv('sp500_joined_closes.csv')





compile_data()