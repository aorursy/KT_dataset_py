# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib as plt
# Any results you write to the current directory are saved as output.
import seaborn as sns
def creating_dataframe():
    df = pd.read_csv("/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")
    # df.info()
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
    df['Date'] = df['Date'].dt.normalize()
    df['Day'] = df['Date'].dt.day_name()
    df['Weighted_Price'] = np.around(df['Weighted_Price'], decimals=2)
    df['Date'] = df['Date'].drop_duplicates(keep='first')
    df['Daily_Change'] = df['Close'].pct_change(fill_method='ffill') # I choose 'Close' value of prices but either 'Open' and 'Weighted_Price' can selected.
    df = df.sort_values(by=['Date'], ascending=True)
    df = df.dropna()
    df['Daily_Change'] = df['Daily_Change'].abs()               # We need changing percentage and it's not important positive or negative way.
    df = df.sort_values(by=['Daily_Change'], ascending=False)
    df = df.reset_index(drop=True)
    return df
creating_dataframe()
def visualization(dataframe):
    sns.set(style="darkgrid")
    w = dataframe.head(300)
    sns.set(rc={'figure.figsize': (12, 8)})
    ax = sns.countplot(x="Day", data=w, )
    return ax

visualization(creating_dataframe())