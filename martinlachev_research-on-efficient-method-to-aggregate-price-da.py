%matplotlib inline
import yfinance as yf
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 10]
data = yf.download('ETH-USD','2019-01-01','2020-01-01')
data.Close.plot()

plt.show()
import numpy as np
import pandas as pd
from datetime import datetime

data = pd.read_csv('data/trade_20190905.csv')
data = data.append(pd.read_csv('data/trade_20190906.csv')) 
data = data.append(pd.read_csv('data/trade_20190907.csv'))

data = data[data.symbol == 'ETHUSD']
data['timestamp'] = data.timestamp.map(lambda t: datetime.strptime(t[:-3], "%Y-%m-%dD%H:%M:%S.%f"))

def compute_vwap(df):
    q = df['foreignNotional']
    p = df['price']
    vwap = np.sum(p * q) / np.sum(q)
    df['vwap'] = vwap
    return df
data_timeidx = data.set_index('timestamp')
data_time_grp = data_timeidx.groupby(pd.Grouper(freq='15Min'))
num_time_bars = len(data_time_grp)
data_time_vwap = data_time_grp.apply(compute_vwap)

total_ticks = len(data)
num_ticks_per_bar = total_ticks / num_time_bars
num_ticks_per_bar = round(num_ticks_per_bar, -3) # round to the nearest thousand
data_tick_grp = data.reset_index().assign(grpId=lambda row: row.index // num_ticks_per_bar)
data_tick_vwap =  data_tick_grp.groupby('grpId').apply(compute_vwap)
data_tick_vwap.set_index('timestamp', inplace=True)

plt.plot(data_time_vwap.vwap)
plt.plot(data_tick_vwap.vwap)
plt.show()
data_cm_vol = data.assign(cmVol=data['homeNotional'].cumsum()) 
total_vol = data_cm_vol.cmVol.values[-1]
vol_per_bar = total_vol / num_time_bars
vol_per_bar = round(vol_per_bar, -2) # round to the nearest hundred
data_vol_grp = data_cm_vol.assign(grpId=lambda row: row.cmVol // vol_per_bar)
data_vol_vwap =  data_vol_grp.groupby('grpId').apply(compute_vwap)
data_vol_vwap.set_index('timestamp', inplace=True)

plt.plot(data_time_vwap.vwap)
plt.plot(data_vol_vwap.vwap)
plt.show()


data_cm_dol = data.assign(cmDol=data['foreignNotional'].cumsum()) 
total_dol = data_cm_dol.cmDol.values[-1]
dol_per_bar = total_dol / num_time_bars
dol_per_bar = round(dol_per_bar, -2) # round to the nearest hundred
data_dol_grp = data_cm_dol.assign(grpId=lambda row: row.cmDol // dol_per_bar)
data_dol_vwap =  data_dol_grp.groupby('grpId').apply(compute_vwap)
data_dol_vwap.set_index('timestamp', inplace=True)

plt.plot(data_time_vwap.vwap)
plt.plot(data_dol_vwap.vwap)
plt.show()