# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#graph library import

import matplotlib.pyplot as plt

from matplotlib import dates as mdates

from matplotlib import ticker as mticker

from matplotlib.finance import candlestick_ohlc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
fname = '../input/prices.csv'

df = pd.read_csv(fname)



df['date'] = pd.to_datetime(df.date)

df.set_index('date').resample('D').sum()



fig = plt.figure()

ax1 = plt.subplot2grid((1,1),(0,0))

plt.ylabel('Price')

ax1.xaxis.set_major_locator(mticker.MaxNLocator(6))

df

#ax1.xaxis.set_major_formatter(df['date'].dt.strftime('%Y-%m-%d'))



#candlestick_ohlc(ax1,[df.open,df.close,df.low,df.close],width=0.2)