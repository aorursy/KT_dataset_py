import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
df = pd.read_csv("../input/AAPL_5year.csv", index_col=0)
df.shape
df.head()
df = df.loc[:, "Close"]

df = pd.DataFrame(df)
df.head()
df.index.name = None

df.head()
df['Close'].plot(figsize=(12,6))
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['Close'], freq=12)

fig = plt.figure()

fig = decomposition.plot()

fig.set_size_inches(12,12)
from statsmodels.tsa.stattools import adfuller



def test_stationarity(ts):

    

    # determining rolling statistics

    rol_mean = ts.rolling(50).mean()

    rol_std = ts.rolling(50).std()

    

    # plotting rolling statistics

    fig = plt.figure(figsize=(12,8))

    orig = plt.plot(ts, color='red', label='original')

    mean = plt.plot(rol_mean, color='blue', label='rolling mean')

    std = plt.plot(rol_std, color='black', label='rolling std')

    plt.legend(loc='best')

    plt.title('rolling mean and standard deviation wrt original data')

    plt.show()

    

    # perform dickey-fuller test to find whether the timeseries is stationary or not

    print("result of dickey-fuller test")

    dftest = adfuller(ts, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '#lags used', '#observation used'])

    for key, value in dftest[4].items():

        dfoutput['critical value (%s)' %key] = value

    print(dfoutput)
test_stationarity(df['Close'])
df['first_diff'] = df['Close'] - df['Close'].shift(1)

test_stationarity(df['first_diff'].dropna(inplace=False))
df.head()
df['seasonal_diff'] = df['Close'] - df['Close'].shift(250)

test_stationarity(df['seasonal_diff'].dropna(inplace=False))
df.head()
df['seasonal_first_diff'] = df['first_diff'] - df['first_diff'].shift(250)

test_stationarity(df['seasonal_first_diff'].dropna(inplace=False))
df.head()
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf



fig = plt.figure(figsize=(12,8))



ax1 = fig.add_subplot(211)

fig = plot_acf(df['seasonal_first_diff'].iloc[251:], lags=40, ax=ax1)



ax2 = fig.add_subplot(212)

fig = plot_pacf(df['seasonal_first_diff'].iloc[251:], lags=40, ax=ax2)
df.head()
df.shape
import warnings

warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
# x = df['Close'].values

# train = x[0:1000]

# test = x[1000:]



# order = (0,1,0)

# seas_order = (1,1,1,100)



# sarima_model = SARIMAX(train, order=order, seasonal_order=seas_order)

# sarima_fit = sarima_model.fit()

# print(sarima_fit.aic)



# sarima_pred = sarima_fit.forecast(steps=259)



# b = pd.DataFrame(columns=['all','predicted'])

# b['all'] = x

# b['predicted'][1000:] = sarima_pred

# b[['all','predicted']].plot(figsize=(12,6))
train = df['Close'][0:1200]



order = (0,1,0)

seas_order = (1,1,1,100)



sarima_model = SARIMAX(train, order=order, seasonal_order=seas_order)

sarima_fit = sarima_model.fit()

print(sarima_fit.aic)



pred = sarima_fit.forecast(steps=59)

df['forecast'] = np.nan

df['forecast'][1200:] = pred

df[['Close', 'forecast']].plot(figsize=(12,6))
plt.plot(df['Close'][1200:])

plt.plot(df['forecast'][1200:], color='red')