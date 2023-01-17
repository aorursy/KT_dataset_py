import requests

from datetime import datetime



import pandas as pd
API_BASE = 'https://api.binance.com/api/v3/'



LABELS = [

    'open_time',

    'open',

    'high',

    'low',

    'close',

    'volume',

    'close_time',

    'quote_asset_volume',

    'number_of_trades',

    'taker_buy_base_asset_volume',

    'taker_buy_quote_asset_volume',

    'ignore'

]
params = {

    'symbol': 'BTCUSDT',

    'interval': '1m',

    'startTime': int(datetime(2019, 3, 12).timestamp() * 1000),

    'limit': 1440

}



response = requests.get(f'{API_BASE}klines', params)

df = pd.DataFrame(response.json(), columns=LABELS)

df.index = df['open_time'].apply(lambda x: datetime.fromtimestamp(x / 1000))

df.drop(['open_time', 'close_time'], axis=1).apply(pd.to_numeric).plot()