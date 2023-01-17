# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.
from collections import namedtuple
FileInfo = namedtuple("FileInfo", ["is_etf", "is_stock", "symbol", "region"])

def get_file_info(path, filename):
    dirname = os.path.split(path)[1]
    if dirname == "ETFs":
        is_etf = True
    elif dirname == "Stocks":
        is_etf = False
    else:
        return None
    
    is_stock = not is_etf
    symbol, region, _ = filename.split(".")
    return FileInfo(is_etf=is_etf, is_stock=is_stock, symbol=symbol, region=region)
    

stocks_info = {}
etfs_info = {}

stocks = {}
etfs = {}
for dirname, _, filenames in os.walk('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data'):
    for filename in filenames:
        file_info = get_file_info(dirname, filename)
        if file_info.is_etf:
            goes_to = etfs_info
            data_goes_to = etfs
        else:
            goes_to = stocks_info
            data_goes_to = stocks
            
        try:
            data_goes_to[file_info.symbol] = pd.read_csv(os.path.join(dirname, filename))
            goes_to[file_info.symbol] = file_info
        except pd.errors.EmptyDataError:
            print("Warning: Could not parse file %s" % os.path.join(dirname, filename))

# check if the symbols are also unique across stocks and etfs
if not stocks.keys().isdisjoint(etfs.keys()):
    raise RuntimeWarning("ETF and stock found sharing the same symbol: " + str(set(stocks.keys()).intersection(etfs.keys())))

info = stocks_info.copy()
info.update(etfs_info)

data = stocks.copy()
data.update(etfs)
    
# Explore the datasets
print(stocks["goog"])
print("Traded volume: %d" % np.sum(stocks["goog"]["Volume"] * stocks["goog"]["Close"]))

print(stocks["goog"][list(map(lambda s: s.startswith("2017"), stocks["goog"]["Date"]))])
num_symbols = 500

# find the most traded stocks
volume = {}
for key in stocks.keys():
    traded_recently = stocks[key][list(map(lambda s: s.startswith("2017"), stocks[key]["Date"]))]
    volume[key] = np.sum(traded_recently["Volume"] * traded_recently["Close"])

symbols = sorted(stocks.keys(), key=lambda key: volume[key], reverse=True)
symbols = symbols[:num_symbols]

print("Using the most traded %d symbols out of %d available symbols" % (len(symbols), len(stocks.keys())))

frames = []
for symbol in symbols:
    symbol_df = stocks[symbol]
    symbol_df = symbol_df[["Date", "Close"]]
    symbol_df.columns = ("Date", symbol)
    symbol_df.set_index("Date", inplace=True)
    frames.append(symbol_df)

    
print("Joining ... This may take some time")
df_joined = pd.concat(frames, axis=1, sort=False)
df_joined.sort_index(inplace=True)

df_pct = df_joined.pct_change()
df_normalized = (df_pct - df_pct.mean()) / df_pct.std()

print(df_normalized)
print("Done!")
offset = 5
show_best = 5
min_periods = 300

df_pct_shifted = df_pct.shift(offset)
df_pct_shifted.columns = [c + "-" + str(offset) for c in df_pct_shifted.columns]

df_full = pd.concat([df_pct, df_pct_shifted], axis=1)

mean = df_full.mean()
second_moment = (df_full ** 2).mean()
cov = df_full.cov(min_periods=min_periods)

tmp = pd.DataFrame(data=np.tensordot(mean.values, mean.values, axes=0), index=mean.index, columns=mean.index)

a = (cov + tmp) / second_moment

err = (cov + tmp) ** 2 
err = err.div(second_moment, axis="columns")
err = err.sub(second_moment, axis="index")
err *= -1

old_symbols = df_pct_shifted.columns

best_guess = []

for symbol in symbols:
    errors = err.loc[symbol][old_symbols].sort_values()[:show_best]
    coeff = a.loc[symbol][errors.index]
    
    errors.name = "MSE"
    coeff.name = "COEFF for {}".format(symbol)
    #print(pd.concat([errors, coeff], axis=1))
    
    best_guess.append((symbol, coeff[errors.index[0]], errors.index[0].replace("-{}".format(offset), ""), errors[0]))

best_guess = sorted(best_guess, key=lambda guess: float("inf") if math.isnan(guess[-1]) else guess[-1])
    
print("\n".join("{}(t) = {} * {}(t-{offset}) with MSE = {}".format(*guess, offset=offset) for guess in best_guess))

