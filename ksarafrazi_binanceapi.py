import requests        # for making http requests to binance

import json            # for parsing what binance sends back to us

import pandas as pd    # for storing and manipulating the data we get back

import numpy as np     # numerical python, i usually need this somewhere 

                       # and so i import by habit nowadays



import matplotlib.pyplot as plt # for charts and such

    

import datetime as dt  # for dealing with times



def get_bars(symbol, interval = '1d'):

    root_url = 'https://api.binance.com/api/v1/klines'

    url = root_url + '?symbol=' + symbol + '&interval=' + interval +'&limit=1000' 

    data = json.loads(requests.get(url).text)

    try:

        df = pd.DataFrame(data)

        df.columns = ['open_time',

                 'open', 'high', 'low', 'close', 'volume',

                 'close_time', 'qav', 'num_trades',

                 'taker_base_vol', 'taker_quote_vol', 'ignore']

        df['close_time'] = [dt.datetime.fromtimestamp(x/1000.0) for x in df.close_time]

        df['date'] = [x.strftime('%Y.%m.%d') for x in df.close_time]

        df['time'] = [x.strftime('%H%M%S') for x in df.close_time]



    

        df.drop(['open_time','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'], axis = 1, inplace = True)

        if symbol == 'XRPBTC':

            df = df[df['low'].astype('float') > 0.0000001]

            

        df.to_csv(symbol + '.csv' , index = False, header = False)

    except:

        print(f"{symbol} not retrieved.")

        df = pd.DataFrame()

            

    return df



ticker_list = ['BTCUSDT','ETHUSDT','TRXBTC','XRPBTC','LTCBTC','EOSBTC','ADABTC','ETHBTC']

#ticker_list = ['THETABTC','LENDBTC']



for ticker in ticker_list:

    data = get_bars(ticker)



#btcusd = get_bars('BTCUSDT')

#xrpbtc = get_bars('XRPBTC')





#close = btcusd['c'].astype('float')

#close.plot(figsize=(16,9))