# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import statsmodels.api as sm
import datetime as dt
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

'''Merging all the time-series into one database'''
hp_df = pd.read_csv('../input/NC-Charlotte.csv')
hp_df.rename(columns = {hp_df.columns[1]:'NC-Charlotte'}, inplace = True)

for file_name in os.listdir("../input")[1:]:
    temp_df = pd.read_csv('../input/' + file_name)
    temp_df.rename(columns = {temp_df.columns[1] : file_name[:-4]}, inplace = True)
    hp_df = hp_df.merge(temp_df, how='outer', on='DATE')
    
hp_df.rename(columns = {'DATE':'date'}, inplace=True)

'''transform "date" column into datetime'''
hp_df['date'] = hp_df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
hp_df.set_index('date', drop = False, inplace = True)
'''Plotting all the time serieses'''
hp_df.plot(x='date', y=hp_df.columns[1:], kind='line', figsize=(20,10))
'''For now I will take only one time-series (i.e. IL-Chicago) and look at it'''
'''Checking the Autocorrelation and Partial Autocorrelation:'''
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(hp_df['IL-Chicago'].values.squeeze(), lags=12, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(hp_df['IL-Chicago'], lags=12, ax=ax2)
'''Fiting the data into ARMA(2,2) model'''
arma_mod22 = sm.tsa.ARMA(hp_df['IL-Chicago'], (4,2)).fit(disp=False, )

print(arma_mod22.params)
'''Plotting the forecast'''
fig, ax = plt.subplots(figsize=(12, 8))
ax = hp_df[hp_df['date'] > dt.date(2015, 1, 1)]['IL-Chicago'].plot(ax=ax)
fig = arma_mod22.plot_predict(dt.datetime(2017,12,1), dt.date(2019, 1, 1),
                              dynamic=True, ax=ax, plot_insample=False, alpha = 0.05)
'''Notes:
1) use just ARMA
2) use ARMA + take data from other indices at time t to ptredict t+1'''
hp_df.head()
