# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



def get_data(stock):

    data = pd.read_csv('/kaggle/input/trading-indexes-apr-2019-to-apr-2020/'+stock+'.csv', index_col='Date', parse_dates=['Date'])

    return data



def calc_corr(ser1, ser2, window):

    ret1 = ser1.pct_change()

    ret2 = ser2.pct_change()

    corr = ret1.rolling(window).corr(ret2)

    return corr



points_to_plot = 300

nasdaq = get_data('nasdaq')

dowjones = get_data('dowjones')

snp500 = get_data('snp500')

data = nasdaq

data['DJ'] = dowjones['High']

data['SP'] = snp500['High']

data['NDX'] = nasdaq['High']

data['rel_str'] = data['NDX'] / data['SP']

data['rel-str2'] = data['NDX'] / data['DJ']



# Calculate 50 day rolling correlation

data['corr1'] = calc_corr(data['NDX'], data['SP'], 100)

data['corr2'] = calc_corr(data['NDX'], data['DJ'], 100)

data.tail(20)

#After this, we slice the data, effectively discarding all but the last 300 data points, using the slicing logic from before.

# Slice the data, cut points we don't intend to plot.

plot_data = data[-points_to_plot:]



# Make  new figure and set the size.

fig = plt.figure(figsize=(12, 8))



# The first subplot, planning for 3 plots high, 1 plot wide, this being the first.

ax = fig.add_subplot(311)

ax.set_title('Index Comparison')

ax.semilogy(plot_data['SP'], linestyle='-', label='S&P 500', linewidth=3.0)

ax.semilogy(plot_data['NDX'], linestyle='--', label='Nasdaq', linewidth=3.0)

ax.legend()

ax.grid(False)



# Second sub plot.

ax = fig.add_subplot(312)

ax.plot(plot_data['rel_str'], label='Relative Strength, Nasdaq to S&P 500', linestyle=':', linewidth=3.0)

ax.legend()

ax.grid(True)



# Third subplot.

ax = fig.add_subplot(313)

ax.plot(plot_data['corr1'], label='Correlation between Nasdaq and S&P 500', linestyle='-.', linewidth=3.0)

ax.legend()

ax.grid(True)
