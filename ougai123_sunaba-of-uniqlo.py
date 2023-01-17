import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt



from numba import autojit

from statsmodels.tsa import arima_model
train = pd.read_csv("../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv")

train = train[::-1]

train["Close"].plot()

train_sample = pd.DataFrame({"Close":train["Close"]})

train_sample.index = train["Date"]

train_sample
@autojit

def option_search():

    diff = np.array((train_sample["Close"] - train_sample["Close"].shift()).dropna())

    print(diff[0])

    # 差分系列への自動ARMA推定関数の実行

    resDiff = sm.tsa.arma_order_select_ic(diff, ic='aic', trend='nc')

    print(resDiff.aic_min_order)
@autojit

def view_diff(data):

    ts_acf = sm.tsa.stattools.acf(data, nlags=40)

    print(ts_acf)

 

# 偏自己相関

    ts_pacf = sm.tsa.stattools.pacf(data, nlags=40, method='ols')

    print(ts_pacf)
from matplotlib import pylab as plt

import seaborn as sns

%matplotlib inline



data = train["Close"].values

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(data, lags=40, ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
option_search()
data = np.array(train_sample.interpolate(method='linear').dropna(), dtype=float)

results=arima_model.ARIMA(data, order = [1,0,1]).fit()



plt.clf()

train_sample.plot()

plt.plot(results.predict(start=1,end=len(data)))

plt.legend(['data','predicted'])

plt.show()
results.predict(start=1,end=len(data))