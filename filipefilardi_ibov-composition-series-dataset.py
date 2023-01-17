!pip install alpha_vantage
import time

import numpy as np

import pandas as pd



from tqdm.notebook import tqdm

from kaggle_secrets import UserSecretsClient

from alpha_vantage.timeseries import TimeSeries
MAX_API_CALL_PER_MINUTE = 5

TIME_BETWEEN_CALLS = 60

API_KEY = UserSecretsClient().get_secret('API_KEY')



ts = TimeSeries(key=API_KEY, output_format='pandas', indexing_type='date')
df_ibov_composition = pd.read_csv('../input/bovespa-index-ibovespa-stocks-data/ibov_composition.csv')
df_ibov_composition['code'].unique()
for i, ticker in enumerate(tqdm(df_ibov_composition['code'].unique()), 1):

    try:

        # Call API to get daily historical data

        df_series, _ = ts.get_daily(symbol=f'{ticker}.SA', outputsize='full')



        # Rename columns

        df_series.rename(columns={

            '1. open' : 'open',

            '2. high' : 'high',

            '3. low' : 'low',

            '4. close' : 'close',

            '5. volume' : 'volume',

        }, inplace=True)



        # Save Series

        df_series.sort_index().to_csv(f'{ticker}_series.csv')

    except:

        print(f' Something went wrong with {ticker}')

    

    if i % MAX_API_CALL_PER_MINUTE == 0:

        time.sleep(TIME_BETWEEN_CALLS)