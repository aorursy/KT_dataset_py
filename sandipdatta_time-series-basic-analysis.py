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
# Import Packages

import warnings

warnings.filterwarnings('ignore')

import numpy as np 

import pandas as pd 

from pandas import DataFrame

import matplotlib.pyplot as plt



#importing packages for the prediction of time-series data

from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm

import statsmodels.tsa.api as smt

import statsmodels.formula.api as smf



from sklearn.metrics import mean_squared_error



%matplotlib inline
df = pd.read_csv('../input/bitcoin_price.csv', parse_dates=['Date'])

df.head(3)
print (df.describe())

print ("=============================================================")

print (df.dtypes)
df1 = df[['Date','Close']]

df1.head(3)
# Setting the Date as Index

df_ts = df1.set_index('Date')

df_ts.sort_index(inplace=True)

print (type(df_ts))

print (df_ts.head(3))

print ("========================")

print (df_ts.tail(3))
# Basic plot 

df_ts.plot()
# Dickey Fuller Test Function

def test_stationarity(timeseries):

    # Perform Dickey-Fuller test:

    from statsmodels.tsa.stattools import adfuller

    print('Results of Dickey-Fuller Test:')

    print ("==============================================")

    

    dftest = adfuller(timeseries, autolag='AIC')

    

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])

    

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    

    print(dfoutput)

    
# Stationarity Check - Lets do a quick check on Stationarity with Dickey Fuller Test 

# Convert the DF to series first

ts = df_ts['Close']

test_stationarity(ts)

# Let's plot the 12-Month Moving Rolling Mean & Variance and find Insights

# Rolling Statistics

rolmean = ts.rolling(window=12).mean()

rolvar = ts.rolling(window=12).std()



plt.plot(ts, label='Original')

plt.plot(rolmean, label='Rolling Mean')

plt.plot(rolvar, label='Rolling Standard Variance')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False)
# Lets do a quick vanila decomposition to see any trend seasonality etc in the ts

decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative')



fig = decomposition.plot()

fig.set_figwidth(12)

fig.set_figheight(8)

fig.suptitle('Decomposition of multiplicative time series')

plt.show()
# Lets Resample the data by Month and analyze again

df_ts_m = df_ts.resample('M').mean()

print (type(df_ts_m))

print (df_ts_m.head(3))
tsm = df_ts_m['Close']

print (type(tsm))
# Stationarity Check

test_stationarity(tsm)
# Lets do a quick vanila decomposition to see any trend seasonality etc in the ts

decomposition = sm.tsa.seasonal_decompose(tsm, model='multiplicative')



fig = decomposition.plot()

fig.set_figwidth(12)

fig.set_figheight(8)

fig.suptitle('Decomposition of multiplicative time series')

plt.show()
# lets try to make the "tsm" Stationary



tsmlog = np.log10(tsm)

tsmlog.dropna(inplace=True)



tsmlogdiff = tsmlog.diff(periods=1)

tsmlogdiff.dropna(inplace=True)

# Stationarity Check

test_stationarity(tsmlogdiff)
# Let's plot ACF & PACF graphs to visualize AR & MA components



fig, axes = plt.subplots(1, 2)

fig.set_figwidth(12)

fig.set_figheight(4)

smt.graphics.plot_acf(tsmlogdiff, lags=30, ax=axes[0], alpha=0.5)

smt.graphics.plot_pacf(tsmlogdiff, lags=30, ax=axes[1], alpha=0.5)

plt.tight_layout()