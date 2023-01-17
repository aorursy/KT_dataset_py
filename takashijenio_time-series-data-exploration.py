import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn; seaborn.set()

%matplotlib inline



# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/bitcoin_price_Training - Training.csv")

test = pd.read_csv("../input/bitcoin_price_1week_Test - Test.csv")
print(train.shape)

print(test.shape)
train.head()
train.tail()
test
train = train[::-1] 

test = test[::-1]

train.head()
from dateutil.parser import parse

from datetime import datetime



def convert(date):

    holder = []

    for i in date:

        tp = parse(i).timestamp()

        dt = datetime.fromtimestamp(tp)

        holder.append(dt)

    return np.array(holder)
date = train['Date'].values

date_n = convert(date)
# sanity check

print(len(date_n) == train.shape[0])
train['Date'] = date_n

train.head()
train = train.set_index('Date')

train.head()
train.describe()
# check the missing values

train.isnull().any()
plt.figure(num=None, figsize=(20, 6))

plt.subplot(1,2,1)

ax = train['Close'].plot(style=['-'])

ax.lines[0].set_alpha(0.3)

ax.set_ylim(0, np.max(train['Close'] + 100))

plt.xticks(rotation=90)

plt.title("No scaling")

ax.legend()

plt.subplot(1,2,2)

ax = train['Close'].plot(style=['-'])

ax.lines[0].set_alpha(0.3)

ax.set_yscale('log')

ax.set_ylim(0, np.max(train['Close'] + 100))

plt.xticks(rotation=90)

plt.title("logarithmic scale")

ax.legend()
close = train['Close']

close.plot(alpha=0.5, style='-')

close.resample('BA').mean().plot(style=':')

close.asfreq('BA').plot(style='--')

plt.yscale('log')

plt.title("logarithmic scale")

plt.legend(['close-price', 'resample', 'asfreq'], 

           loc='upper left')

# 'resample'-- average of the previous year

# 'asfreq' -- value at the end of the year
ROI = 100 * (close.tshift(-365) / close - 1)

ROI.plot()

plt.ylabel('% Return on Investment');
rolling = close.rolling(200, center=True)



data = pd.DataFrame({'input': close, 

                     '200days rolling_mean': rolling.mean(), 

                     '200days rolling_std': rolling.std()})



ax = data.plot(style=['-', '--', ':'])

ax.set_yscale('log')

ax.set_title("SMA on log scale")

rolling = close.rolling(365, center=True)

ax.lines[0].set_alpha(0.3)
ax = data.plot(style=['-', '--', ':'])

ax.set_title("SMA on raw data")

ax.lines[0].set_alpha(0.3)
rolling = pd.ewma(close, com=200)



data = pd.DataFrame({'input': close, 

                     '200days rolling_mean': rolling.mean(), 

                     '200days rolling_std': rolling.std()})



ax = data.plot(style=['-', '--', ':'])

ax.set_yscale('log')

ax.set_title("EMA on log scale")

ax.lines[0].set_alpha(0.3)
ax = data.plot(style=['-', '--', ':'])

ax.set_title("EMA on raw data")

ax.lines[0].set_alpha(0.3)
from pandas.plotting import lag_plot

lag_plot(close)
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(close)
from pandas import Series

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(close, lags=50)
from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error



test = test['Close'].values
train_pr = train['Close'].values
# train and fit autoregression

model = AR(train_pr)

model_fit = model.fit()



print("Lag: %s" % model_fit.k_ar)

print("Coefficients: %s" % model_fit.params)



pred = model_fit.predict(start=len(train), end=len(train_pr)+len(test)-1, dynamic=False)

mse = mean_squared_error(test, pred)

print("Test MSE {0:.3f}".format(mse))
plt.plot(test, label='true value')

plt.plot(pred, color='red', label='prediction')

plt.title("Autoregressive model")

plt.legend()