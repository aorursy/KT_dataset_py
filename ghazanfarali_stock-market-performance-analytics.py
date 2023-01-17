# Importing libraries

import os

import warnings

warnings.filterwarnings('ignore')

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') 

# Above is a special style template for matplotlib, highly useful for visualizing time series data

%matplotlib inline

from pylab import rcParams

#from plotly import tools

#import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import statsmodels.api as sm

from numpy.random import normal, seed

from scipy.stats import norm

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA

import math

from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
FFC = pd.read_csv('../input/ksedataset/FFC.csv', index_col='Date', parse_dates=['Date'])

FFC.head()
FFC = FFC.iloc[1:]

FFC = FFC.fillna(method='ffill')

FFC.head()
FFC['2003':'2019'].plot(subplots=True, figsize=(10,12))

plt.title('FFC stock attributes from 2003 to 2019')

plt.savefig('FFC stocks.png')

plt.show()
FFC['Change'] = FFC.High.div(FFC.High.shift())

FFC['Change'].plot(figsize=(20,8))
FFC['Return'] = FFC.Change.sub(1).mul(100)

FFC['Return'].plot(figsize=(20,8))
FFC.High.pct_change().mul(100).plot(figsize=(20,6)) # Another way to calculate returns
FFC.High.diff().plot(figsize=(20,6))
# We choose ENGRO stocks to compare them with FFC

ENGRO = pd.read_csv('../input/ksedataset/ENGRO.csv', index_col='Date', parse_dates=['Date'])
#ENGRO = ENGRO.iloc[1:]

#ENGRO = ENGRO.fillna(method='ffill')

#ENGRO.head()
# Plotting before normalization

FFC.High.plot()

ENGRO.High.plot()

plt.legend(['FFC','ENGRO'])

plt.show()
# Normalizing and comparison

# Both stocks start from 100

normalized_FFC = FFC.High.div(FFC.High.iloc[0]).mul(100)

normalized_ENGRO = ENGRO.High.div(ENGRO.High.iloc[0]).mul(100)

normalized_FFC.plot()

normalized_ENGRO.plot()

plt.legend(['FFC','ENGRO'])

plt.show()
# Rolling window functions

rolling_FFC = FFC.High.rolling('90D').mean()

FFC.High.plot()

rolling_FFC.plot()

plt.legend(['High','Rolling Mean'])

# Plotting a rolling mean of 90 day window with original High attribute of google stocks

plt.show()
# Expanding window functions

ENGRO_mean = ENGRO.High.expanding().mean()

ENGRO_std = ENGRO.High.expanding().std()

ENGRO.High.plot()

ENGRO_mean.plot()

ENGRO_std.plot()

plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])

plt.show()
# OHLC chart of June 2003

trace = go.Ohlc(x=FFC['06-2003'].index,

                open=FFC['06-2003'].Open,

                high=FFC['06-2003'].High,

                low=FFC['06-2003'].Low,

                close=FFC['06-2003'].Close)

data = [trace]

iplot(data, filename='simple_ohlc')
# OHLC chart of 2003

trace = go.Ohlc(x=FFC['2003'].index,

                open=FFC['2003'].Open,

                high=FFC['2003'].High,

                low=FFC['2003'].Low,

                close=FFC['2003'].Close)

data = [trace]

iplot(data, filename='simple_ohlc')
# OHLC chart of 2003

trace = go.Ohlc(x=FFC.index,

                open=FFC.Open,

                high=FFC.High,

                low=FFC.Low,

                close=FFC.Close)

data = [trace]

iplot(data, filename='simple_ohlc')
# Candlestick chart of march 2003

trace = go.Candlestick(x=FFC['03-2003'].index,

                open=FFC['03-2003'].Open,

                high=FFC['03-2003'].High,

                low=FFC['03-2003'].Low,

                close=FFC['03-2003'].Close)

data = [trace]

iplot(data, filename='simple_candlestick')
# Candlestick chart of 2003

trace = go.Candlestick(x=FFC['2003'].index,

                open=FFC['2003'].Open,

                high=FFC['2003'].High,

                low=FFC['2003'].Low,

                close=FFC['2003'].Close)

data = [trace]

iplot(data, filename='simple_candlestick')
# Candlestick chart of 2003-2019

trace = go.Candlestick(x=FFC.index,

                open=FFC.Open,

                high=FFC.High,

                low=FFC.Low,

                close=FFC.Close)

data = [trace]

iplot(data, filename='simple_candlestick')
# Autocorrelation of FFC of Close

plot_acf(FFC["Close"],lags=25,title="FFC")

plt.show()
# Partial Autocorrelation of closing price of microsoft stocks

plot_pacf(ENGRO["Close"],lags=25)

plt.show()
# Let's take FFC stocks High for this

FFC["High"].plot(figsize=(16,8))
# Now, for decomposition...

rcParams['figure.figsize'] = 11, 9

decomposed_FFC_volume = sm.tsa.seasonal_decompose(FFC["High"],freq=360) # The frequncy is annual

figure = decomposed_FFC_volume.plot()

plt.show()
# Plotting white noise

rcParams['figure.figsize'] = 16, 6

white_noise = np.random.normal(loc=0, scale=1, size=1000)

# loc is mean, scale is variance

plt.plot(white_noise)
# Plotting autocorrelation of white noise

plot_acf(white_noise,lags=20)

plt.show()
# Augmented Dickey-Fuller test on volume of FFC and ENGRO stocks 

adf = adfuller(ENGRO["Volume"])

print("p-value of ENGRO: {}".format(float(adf[1])))

adf = adfuller(FFC["Volume"])

print("p-value of FFC: {}".format(float(adf[1])))
seed(42)

rcParams['figure.figsize'] = 16, 6

random_walk = normal(loc=0, scale=0.01, size=1000)

plt.plot(random_walk)

plt.show()
fig = ff.create_distplot([random_walk],['Random Walk'],bin_size=0.001)

iplot(fig, filename='Basic Distplot')
# The original non-stationary plot

decomposed_FFC_volume.trend.plot()
# The new stationary plot

decomposed_FFC_volume.trend.diff().plot()
# AR(1) MA(1) model:AR parameter = +0.9

rcParams['figure.figsize'] = 16, 12

plt.subplot(4,1,1)

ar1 = np.array([1, -0.9]) # We choose -0.9 as AR parameter is +0.9

ma1 = np.array([1])

AR1 = ArmaProcess(ar1, ma1)

sim1 = AR1.generate_sample(nsample=1000)

plt.title('AR(1) model: AR parameter = +0.9')

plt.plot(sim1)

# We will take care of MA model later

# AR(1) MA(1) AR parameter = -0.9

plt.subplot(4,1,2)

ar2 = np.array([1, 0.9]) # We choose +0.9 as AR parameter is -0.9

ma2 = np.array([1])

AR2 = ArmaProcess(ar2, ma2)

sim2 = AR2.generate_sample(nsample=1000)

plt.title('AR(1) model: AR parameter = -0.9')

plt.plot(sim2)

# AR(2) MA(1) AR parameter = 0.9

plt.subplot(4,1,3)

ar3 = np.array([2, -0.9]) # We choose -0.9 as AR parameter is +0.9

ma3 = np.array([1])

AR3 = ArmaProcess(ar3, ma3)

sim3 = AR3.generate_sample(nsample=1000)

plt.title('AR(2) model: AR parameter = +0.9')

plt.plot(sim3)

# AR(2) MA(1) AR parameter = -0.9

plt.subplot(4,1,4)

ar4 = np.array([2, 0.9]) # We choose +0.9 as AR parameter is -0.9

ma4 = np.array([1])

AR4 = ArmaProcess(ar4, ma4)

sim4 = AR4.generate_sample(nsample=1000)

plt.title('AR(2) model: AR parameter = -0.9')

plt.plot(sim4)

plt.show()
model = ARMA(sim1, order=(1,0))

result = model.fit()

print(result.summary())

print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))

# Predicting simulated AR(1) model 

result.plot_predict(start=900, end=1010)

plt.show()
rmse = math.sqrt(mean_squared_error(sim1[900:1011], result.predict(start=900,end=999)))

print("The root mean squared error is {}.".format(rmse))
# Predicting closing prices of google

humid = ARMA(FFC["Close"].diff().iloc[1:].values, order=(1,0))

res = humid.fit()

res.plot_predict(start=900, end=1010)

plt.show()
rcParams['figure.figsize'] = 16, 6

ar1 = np.array([1])

ma1 = np.array([1, -0.5])

MA1 = ArmaProcess(ar1, ma1)

sim1 = MA1.generate_sample(nsample=1000)

plt.plot(sim1)
model = ARMA(sim1, order=(0,1))

result = model.fit()

print(result.summary())

print("μ={} ,θ={}".format(result.params[0],result.params[1]))
# Forecasting and predicting ENGRO stocks volume

model = ARMA(ENGRO["Volume"].diff().iloc[1:].values, order=(3,3))

result = model.fit()

print(result.summary())

print("μ={}, ϕ={}, θ={}".format(result.params[0],result.params[1],result.params[2]))

result.plot_predict(start=1000, end=1100)

plt.show()
rmse = math.sqrt(mean_squared_error(ENGRO["Volume"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))

print("The root mean squared error is {}.".format(rmse))
# Predicting the ENGRO stocks volume

rcParams['figure.figsize'] = 16, 6

model = ARIMA(ENGRO["Volume"].diff().iloc[1:].values, order=(2,1,0))

result = model.fit()

print(result.summary())

result.plot_predict(start=700, end=1000)

plt.show()
rmse = math.sqrt(mean_squared_error(ENGRO["Volume"].diff().iloc[700:1001].values, result.predict(start=700,end=1000)))

print("The root mean squared error is {}.".format(rmse))
# Predicting closing price of FFC and ENGRO

train_sample = pd.concat([FFC["Close"].diff().iloc[1:],ENGRO["Close"].diff().iloc[1:]],axis=1)

model = sm.tsa.VARMAX(train_sample,order=(2,1),trend='c')

result = model.fit(maxiter=1000,disp=False)

print(result.summary())

predicted_result = result.predict(start=0, end=1000)

result.plot_diagnostics()

# calculating error

rmse = math.sqrt(mean_squared_error(train_sample.iloc[1:1002].values, predicted_result.values))

print("The root mean squared error is {}.".format(rmse))
# Predicting closing price of FFC'

train_sample = FFC["Close"].diff().iloc[1:].values

model = sm.tsa.SARIMAX(train_sample,order=(4,0,4),trend='c')

result = model.fit(maxiter=1000,disp=False)

print(result.summary())

predicted_result = result.predict(start=0, end=500)

result.plot_diagnostics()

# calculating error

rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))

print("The root mean squared error is {}.".format(rmse))
plt.plot(train_sample[1:502],color='red')

plt.plot(predicted_result,color='blue')

plt.legend(['Actual','Predicted'])

plt.title('FFC Closing prices')

plt.show()
# Predicting closing price of FFC'

train_sample = FFC["Close"].diff().iloc[1:].values

model = sm.tsa.UnobservedComponents(train_sample,'local level')

result = model.fit(maxiter=1000,disp=False)

print(result.summary())

predicted_result = result.predict(start=0, end=500)

result.plot_diagnostics()

# calculating error

rmse = math.sqrt(mean_squared_error(train_sample[1:502], predicted_result))

print("The root mean squared error is {}.".format(rmse))
plt.plot(train_sample[1:502],color='red')

plt.plot(predicted_result,color='blue')

plt.legend(['Actual','Predicted'])

plt.title('FFC Closing prices')

plt.show()
# Predicting closing price of FFC and ENGRO

train_sample = pd.concat([FFC["Close"].diff().iloc[1:],ENGRO["Close"].diff().iloc[1:]],axis=1)

model = sm.tsa.DynamicFactor(train_sample, k_factors=1, factor_order=2)

result = model.fit(maxiter=1000,disp=False)

print(result.summary())

predicted_result = result.predict(start=0, end=1000)

result.plot_diagnostics()

# calculating error

rmse = math.sqrt(mean_squared_error(train_sample.iloc[1:1002].values, predicted_result.values))

print("The root mean squared error is {}.".format(rmse))