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
df = pd.read_csv('/kaggle/input/Result_of_backup1.csv')
df.head()
df.columns = ['day', 'sys_start_time', 'backup_size', 'backup_none']
df.head()
df.isna().sum()
df.dtypes
df.sys_start_time = pd.to_datetime(df.sys_start_time)

df.dtypes
df.head()
import matplotlib.pyplot as plt
df.sys_start_time.plot()
plt.show()

df.set_index('sys_start_time', inplace=True)
df.head()
df.drop(['backup_none','day'], inplace = True, axis='columns')
df.head()
df.index = df.index.date
df.head()
df.backup_size.isna().sum()
time_series = df.asfreq('d')
time_series.head()
time_series.backup_size.plot(figsize=(20,5))
df.backup_size.plot(figsize=(20,5))
# If time series follows random walk, future predictions cannot be made

from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as sgt
adfuller(df.backup_size)
# 50% change data comes from non stationary process
sgt.plot_acf(df.backup_size, lags = 40, zero=False)
plt.show()
sgt.plot_pacf(df.backup_size, lags = 40, zero = False, method = ('ols'))
plt.show()