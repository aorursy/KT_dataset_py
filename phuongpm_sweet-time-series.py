# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Data management

import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))



# Visualization

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') #style for time series visualization

%matplotlib inline

from pylab import rcParams

from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



#Statistics 

import statsmodels.api as sm

from numpy.random import normal, seed

from scipy.stats import norm

from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error









# Any results you write to the current directory are saved as output.
candy = pd.read_csv("../input/candy_production.csv")
candy.tail()
candy.info()
# Create time series from pandas

rng = pd.date_range(start = '1/1/1972', end = '8/1/2017', freq = 'MS')
cdy = pd.Series(list(candy['IPG3113N']), index = rng)
rcParams['figure.figsize'] = 20, 5

cdy.plot(linewidth = 1, ls = 'solid')

plt.title('Monthly US Candy Production (1972 - 2017)')

plt.show()
seconds_of_year = 365*24*3600

frac = [0]*len(rng)

for i in range(1,len(rng)):

    frac[i] = round((rng[i] - rng[0]).total_seconds()/seconds_of_year, 3)

frac = np.array(frac).reshape(-1,1)
reg = LinearRegression().fit(frac, cdy)
reg.score(frac, cdy)
lr_detrended = cdy - reg.predict(frac)
lr_detrended.plot(lw = 1)

plt.title('Linear Regression Detrended')

plt.show()
frac_sq = frac**2

X = np.concatenate([frac_sq, frac], axis = 1)
X.shape
reg_2 = LinearRegression().fit(X, cdy)
reg_2.score(X, cdy)
lr_detrended_2 = cdy - reg_2.predict(X)
lr_detrended_2.plot(lw = 1)

plt.title('Linear Regression Detrended (Quadratic Equation)')

plt.show()
mean_year = cdy.resample('Y').mean()
mean_year.head()
dup_mean_year = pd.Series(np.repeat(mean_year.values,12,axis=0)[:548], index = rng)
dup_mean_year.head()
nonlinear_transformed = cdy - dup_mean_year
nonlinear_transformed.plot(lw = 1)

plt.title("Non-linear transformation")

plt.show()
rcParams['figure.figsize'] = 20, 5

diff_1 = cdy.diff()

diff_2 = diff_1.diff()

ax1 = plt.subplot(211)

diff_1.plot(lw = 1, axes = ax1)

ax2 = plt.subplot(212, sharex=ax1)

diff_2.plot(lw = 1, axes = ax2)

ax1.set_title("Differentiation (1st order)")

ax2.set_title("Differentiation (2nd order)")

plt.show()
candy['lr_residual'] = lr_detrended.values
candy['month'] = candy['observation_date'].apply(lambda x: x[5:7])
candy.head()
month_average = candy.groupby('month')['lr_residual'].mean().reset_index().rename(columns = {'lr_residual':'month_average'})

m = list(month_average['month_average'].values)

dup_mean_month = pd.Series((m*(2017-1971))[:548], index = rng)



month_average
subtract_month = lr_detrended - dup_mean_month
rcParams['figure.figsize'] = 20, 5

dup_mean_month.plot(lw = 1)

plt.title("Seasonal Pattern")
rcParams['figure.figsize'] = 20,9

ax1 = plt.subplot(411)

cdy.plot(lw = 1, axes = ax1)

ax1.set_title("Observed")



ax2 = plt.subplot(412, sharex=ax1)

trend = pd.Series(reg.predict(frac), index = rng)

trend.plot(lw = 1, axes=ax2)

ax2.set_title("Trend")



ax3 = plt.subplot(413, sharex=ax1)

dup_mean_month.plot(lw = 1, axes=ax3)

ax3.set_title("Seasonal Pattern")





ax4 = plt.subplot(414, sharex=ax1)

subtract_month.plot(lw = 1, axes=ax4)

ax4.set_title('Seasonal pattern extracted by monthly average (Residual)')

plt.show()
rcParams['figure.figsize'] = 20, 5

# diff_monthly = cdy.diff(periods = 12)

diff_monthly=diff_1.diff(periods=12)

ax1 = plt.subplot(211)

diff_1.plot(lw = 1, axes = ax1)

# ax2 = plt.subplot(312, sharex=ax1)

# diff_2.plot(lw = 1, axes = ax2)

ax3 = plt.subplot(212, sharex=ax1)

diff_monthly.plot(lw = 1, axes = ax3)

ax1.set_title("Differentiation (1st order)")

ax2.set_title("Differentiation (2nd order)")

ax3.set_title("Seasonal differentiation (1st order)")

plt.show()
moving_average = diff_1.rolling(3, center = True).mean()
rcParams['figure.figsize'] = 20, 5

diff_1.plot(lw = 1, alpha = 0.5, label = '1st order diff')

moving_average.plot(lw = 1, color = 'red', label = 'moving average')

plt.title("Removing seasonal patterns by moving average (window size = 3)")

plt.legend()

plt.show()
# Now, for decomposition...

rcParams['figure.figsize'] = 20, 9

decomposed = sm.tsa.seasonal_decompose(cdy,freq=30) # The frequncy is monthly

figure = decomposed.plot()

plt.show()
rcParams['figure.figsize'] = 20, 15

plt.subplots_adjust(hspace=0.5)



ax1 = plt.subplot(611)

plot_acf(cdy, ax = ax1, marker = '.', lags=200)

ax1.set_title("ACF of original series")



ax2 = plt.subplot(612)

plot_acf(lr_detrended, ax = ax2, marker = '.', lags=200)

ax2.set_title("ACF of linear regression detrended")





ax3 = plt.subplot(613)

plot_acf(diff_1.dropna(), ax = ax3, marker = '.', lags = 200)

ax3.set_title("ACF of 1st order differentiation")



ax4 = plt.subplot(614)

plot_acf(moving_average.dropna(), ax=ax4, marker='.', lags=200)

ax4.set_title("ACF of moving average")



ax5 = plt.subplot(615)

plot_acf(decomposed.resid.dropna(), ax=ax5, marker='.', lags=200)

ax5.set_title("ACF of built-in decomposed residual")



ax6 = plt.subplot(616)

plot_acf(diff_monthly.dropna(), ax=ax6, marker='.', lags=200)

ax6.set_title("ACF of monthly differentiation")



plt.show()
white_noise = pd.Series(normal(size=500))
rcParams['figure.figsize'] = 20, 8

plt.subplots_adjust(hspace=0.5)



ax1 = plt.subplot(211)

white_noise.plot(lw=1)

ax1.set_title("White noise series")



ax2 = plt.subplot(212)

plot_acf(white_noise, marker = '.', lags = 200,ax=ax2)

ax2.set_title("ACF of white noise")



plt.show()
# stationary_cdy = decomposed.resid.dropna()

stationary_cdy = diff_monthly.dropna()
# stationary_cdy has data from 1973-04-01 to 2016-05-01 (518 data points)

# split 80-20 for train and test

ind_80 = int(len(stationary_cdy)*0.8)

train, test = stationary_cdy[:ind_80], stationary_cdy[ind_80:]

print(len(train), len(test))
plot_pacf(stationary_cdy, title = 'PACF of the stationary candy series', alpha = 0.05, lags = 100)

plt.ylabel('PACF')

plt.xlabel('Lags')

plt.show()
# Order is determined by BIC

ar_aic = AR(stationary_cdy).fit(ic = 'bic')

print('Lag by BIC: %s' % ar_aic.k_ar)
def rolling_forecast_evaluation(order):

    preds = []

    for i in range(ind_80,len(stationary_cdy)):

        cdy_train = stationary_cdy[:i]

        ar_model = AR(cdy_train).fit(maxlag = order)

        one_ahead_predict = ar_model.predict(start = len(cdy_train), end = len(cdy_train), dynamic = False)

        preds.append(one_ahead_predict[0])



    print("MSE of order = %d: %.3f" % (order,mean_squared_error(test, preds)))
print(">>> Evaluated by rolling forecast")

rolling_forecast_evaluation(18)
# model_13 = AR(stationary_cdy).fit(maxlag = 13)

model_18 = AR(train).fit(maxlag=18)

# preds_13 = model_13.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

preds_18 = model_18.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

print(">>> Evaluated by normal cross validation")

# print("MSE of order = 13: %.3f" % (mean_squared_error(test, preds_13)))

print("MSE of order = %d: %.3f" % (model_18.k_ar,mean_squared_error(test, preds_18)))
rcParams['figure.figsize'] = 20, 8

all_preds=AR(stationary_cdy).fit(maxlag=18).predict(start = 18, end = len(stationary_cdy)-1, dynamic = False)

# First, plot both series on the same plot

ax1 = plt.subplot(311)

stationary_cdy.plot(lw=1, label = 'stationary')

all_preds.plot(lw=1, label = 'AR(18)')

ax1.legend()



residual = (stationary_cdy - all_preds)

ax2 = plt.subplot(312)

residual.plot(lw=1, label = 'residual')

ax2.legend()



ax3 = plt.subplot(313)

plot_pacf(residual, title = None, alpha = 0.05, lags = 100, ax=ax3, label = 'Residual')

ax3.set_ylabel('PACF')

ax3.legend()



plt.show()
rcParams['figure.figsize'] = 20, 8

cdy_model=AR(cdy).fit()

all_preds=cdy_model.predict(start = cdy_model.k_ar, end = len(cdy)-1, dynamic = False)

# First, plot both series on the same plot

ax1 = plt.subplot(311)

cdy.plot(lw=1, label = 'original')

all_preds.plot(lw=1, label = 'AR')

ax1.legend()



residual = (cdy - all_preds)

ax2 = plt.subplot(312)

residual.plot(lw=1, label = 'residual')

ax2.legend()



ax3 = plt.subplot(313)

plot_pacf(residual, title = None, alpha = 0.05, lags = 100, ax=ax3, label = 'Residual')

ax3.set_ylabel('PACF')

ax3.legend()



plt.show()
np.random.seed(42)

random_walk = list()

random_walk.append(-1 if np.random.random() < 0.5 else 1)

for i in range(1, 1000):

    movement = -1 if np.random.random() < 0.5 else 1

    value = random_walk[i-1] + movement

    random_walk.append(value)

    

rcParams['figure.figsize'] = 20, 4

plt.plot(random_walk, lw = 1)

plt.title("Simulation of random walk model")

plt.show()
rcParams['figure.figsize'] = 20, 6



plt.subplots_adjust(hspace = 0.5)

rw_diff = pd.Series(random_walk).diff(periods=1)

ax1 = plt.subplot(211)

rw_diff.plot(lw=1, ax=ax1)

ax1.set_ylim(1.5,-1.5)

ax1.set_title('1st order differencing of random walk model')



ax2 = plt.subplot(212)

plot_acf(rw_diff.dropna(), ax=ax2, lags=200, marker = '.', title = 'ACF of 1st order differencing of random walk model')





plt.show()
plot_acf(stationary_cdy, lags = 50, title = 'Candy production ACF')

plt.show()

plot_pacf(stationary_cdy, lags = 50, title = 'Candy production Partial ACF')

plt.show()
p,q = 12,2

#ARMA model doesn't have a nice information criterion built-in hyperparameters tuning

arma_model = ARMA(stationary_cdy, order = (p,q)).fit(disp=0) 

fig, ax = plt.subplots()

ax = stationary_cdy.plot(ax=ax, label='Candy')

arma_model.plot_predict('2015','2019',ax=ax, plot_insample=False)

plt.show()
rcParams['figure.figsize'] = 20, 6

plot_acf(stationary_cdy, lags = 50, title = 'Candy production ACF')

plt.show()

plot_pacf(stationary_cdy, lags = 50, title = 'Candy production Partial ACF')

plt.show()
model = ARIMA(stationary_cdy, order=[12, 1, 2])

model_fit = model.fit(disp=0)
prediction = model_fit.predict()

fig, ax = plt.subplots()

stationary_cdy.plot(ax=ax, label='Candy production')

# prediction.plot(ax=ax, label='ARIMA Predicted')

model_fit.plot_predict('2015','2019',ax=ax, plot_insample=False)

ax.legend()