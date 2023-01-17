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
df = pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv')



df.head()
from datetime import datetime    # to translate unix time stamp to UTC time
datetime.utcfromtimestamp(1600041360)   # example
df['Timestamp'] = df['Timestamp'].apply(datetime.utcfromtimestamp)



df
df = df.set_index('Timestamp', drop=True)
df
# follow the steps above: fill NA Volume columns with 0.0

df[['Volume_(BTC)','Volume_(Currency)']] = df[['Volume_(BTC)','Volume_(Currency)']].fillna(0)



df
# follow the steps above: fill NA Close columns with PREVIOUS Close price (forward fill)

df['Close'] = df['Close'].fillna(method='ffill')

df
# follow the steps above: fill NA Open, High, Low columns with Close price (row wise backfill)

df = df.fillna(axis=1, method='backfill')

df
import matplotlib.pyplot as plt
plt.plot(df['Close'])
# agg dictionary for OHLCV data

agg_functions = {

    'Open': 'first',

    'High': np.max,

    'Low': np.min,

    'Close': 'last',

    'Volume_(BTC)': np.sum,

    'Volume_(Currency)': np.sum   

}
df_1H = df.resample('1H').agg(agg_functions)

df_1H
df_4H = df.resample('4H').agg(agg_functions)
df_1D = df.resample('1D').agg(agg_functions)
df_1H.to_csv('./BTC_1H.csv')

df_4H.to_csv('./BTC_4H.csv')

df_1D.to_csv('./BTC_1D.csv')