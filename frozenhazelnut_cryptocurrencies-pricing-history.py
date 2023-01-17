# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import packages and get dataset ready

import matplotlib.dates as mdates

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

from datetime import date

from io import StringIO

from pandas.tools.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=15,6

%matplotlib inline



#A few preparation of cryptocurrencies pricing data:

list_of_currency_files = check_output(["ls", "../input"]).decode("utf8").split('\n')

input_folder = '../input/'

list_of_currency_files.remove('')
for file in list_of_currency_files:

    if(file != ''):

        print(file[:-4]+" = pd.read_csv('"+input_folder+file+"')")

        try:

            exec(file[:-4]+" = pd.read_csv('"+input_folder+file+"',parse_dates=['Date'])")

        except Exception as exp:

            print(exp)

            exec(file[:-4]+" = pd.read_csv('"+input_folder+file+"')")
#Bitcoin

color = sns.color_palette()

bitcoin_price['Date_mpl'] = bitcoin_price['Date'].apply(lambda x: mdates.date2num(x))



print(bitcoin_price.head())

fig, ax = plt.subplots(figsize=(10,8))

sns.tsplot(bitcoin_price.Close.values, time=bitcoin_price.Date_mpl.values, alpha=0.8,color = color[2], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.ylabel('Price in USD', fontsize=12)

plt.title("Closing price distribution of bitcoin", fontsize=15)

plt.show()
#Litecoin:

litecoin_price['Date_mpl'] = litecoin_price['Date'].apply(lambda x: mdates.date2num(x))



print(litecoin_price.head())

fig, ax = plt.subplots(figsize=(10,8))

sns.tsplot(litecoin_price.Close.values, time=litecoin_price.Date_mpl.values, alpha=0.8,color = color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.title("Closing price distribution of litecoin", fontsize=15)

plt.show()
#Ripple:

ripple_price['Date_mpl'] = ripple_price['Date'].apply(lambda x: mdates.date2num(x))



print(ripple_price.head())

fig, ax = plt.subplots(figsize=(10,8))

sns.tsplot(ripple_price.Close.values, time=ripple_price.Date_mpl.values, alpha=0.8,color = color[1], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.title("Closing price distribution of litecoin", fontsize=15)

plt.show()
#Monero:



monero_price['Date_mpl'] = monero_price['Date'].apply(lambda x: mdates.date2num(x))



print(litecoin_price.head())

fig, ax = plt.subplots(figsize=(10,8))

sns.tsplot(monero_price.Close.values, time=monero_price.Date_mpl.values, alpha=0.8,color = color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.title("Closing price distribution of litecoin", fontsize=15)

plt.show()
#Ethereum:



ethereum_price['Date_mpl'] = ethereum_price['Date'].apply(lambda x: mdates.date2num(x))



print(litecoin_price.head())

fig, ax = plt.subplots(figsize=(10,8))

sns.tsplot(ethereum_price.Close.values, time=ethereum_price.Date_mpl.values, alpha=0.8,color = color[3], ax=ax)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

fig.autofmt_xdate()

plt.title("Closing price distribution of litecoin", fontsize=15)

plt.show()