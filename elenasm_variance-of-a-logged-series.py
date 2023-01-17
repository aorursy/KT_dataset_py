
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/time-series-starter-dataset/Month_Value_1.csv')
data.head()
print('\n data types:')
data.dtypes
data["Revenue - millions -"] = data["Revenue"]/1000000
data.head()
import matplotlib.dates as mdates

from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(111)

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
x = data['Period']
ts = data['Revenue - millions -'] 
#ax.plot(data['Period'], data['Revenue/10000'])
ax.plot(x, ts)
fig.autofmt_xdate()

ax.grid(True)

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
x = data['Period']
ts_log = np.log(ts) 
#ax.plot(data['Period'], data['Revenue/10000'])
ax.plot(x, ts_log)
fig.autofmt_xdate()

ax.grid(True)

plt.show()
ts_log_df = pd.DataFrame(ts_log)
ts_log_df['Variance'] = 1

for i in ts_log_df.index:
    ts_log_df['Variance'].iloc[i] = ts_log_df['Revenue - millions -'].iloc[0:i].var()
ts_log_df.head(20)
nb = np.arange(0,96)
plt.plot(nb,ts_log_df['Variance'])