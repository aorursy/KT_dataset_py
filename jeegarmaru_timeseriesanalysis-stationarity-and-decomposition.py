import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from statsmodels.tsa import seasonal
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def get_stock_data(symbol):
    df = pd.read_csv("../input/Data/Stocks/{}.us.txt".format(symbol), index_col='Date', parse_dates=True, 
                     na_values='nan', usecols=['Date', 'Close'])
    return df
ibm = get_stock_data('ibm')
ibm.plot(figsize=(15, 5))
def plot_acf(df):
    plt.clf()
    fig, ax = plt.subplots(figsize=(15, 5))
    autocorrelation_plot(df, ax=ax)
def calculate_acf(series, nlags=100):
    alpha = 0.05
    acf_value, confint, qstat, pvalues, *_ = acf(series,
                                             unbiased=True,
                                             nlags=nlags,
                                             qstat=True,
                                             alpha=alpha)
    for l, p_val in enumerate(pvalues):
        if p_val > alpha:
            print("Null hypothesis is accepted at lag = {} for p-val = {}".format(l, p_val))
        else:
            print("Null hypothesis is rejected at lag = {} for p-val = {}".format(l, p_val))
dm = ibm.resample('M').mean()
dm.plot(figsize=(15, 5))
plot_acf(dm)
dmsd = dm.diff(12)[12:]
dmsd.plot()
plot_acf(dmsd)
adf_result_dm = adfuller(dm['Close'])
print ("p-val of the ADF test for monthly data : ", adf_result_dm[1])
adf_result_dmsd = adfuller(dmsd['Close'])
print ("p-val of the ADF test for monthly seasonal differences : ", adf_result_dmsd[1])
def moving_average(sf, window):
    dma = sf.rolling(window=window).mean()
    return dma.loc[~pd.isnull(dma)]
ibm.plot(figsize=(15, 5))
SixXMA6 = moving_average(moving_average(ibm['Close'], window=6), window=6)
TenXMA10 = moving_average(moving_average(ibm['Close'], window=10), window=10)
TwentyXMA20 = moving_average(moving_average(ibm['Close'], window=50), window=50)
f, ax = plt.subplots(4, sharex=True, figsize=(15, 10))

ibm['Close'].plot(color='b', linestyle='-', ax=ax[0])
ax[0].set_title('Raw data')

SixXMA6.plot(color='r', linestyle='-', ax=ax[1])
ax[1].set_title('6x6 day MA')

TenXMA10.plot(color='g', linestyle='-', ax=ax[2])
ax[2].set_title('10x10 day MA')

TwentyXMA20.plot(color='k', linestyle='-', ax=ax[3])
ax[3].set_title('20x20 day MA')
residuals = ibm['Close']-TenXMA10
residuals = residuals.loc[~pd.isnull(residuals)]
residuals.plot(figsize=(15, 5))
plot_acf(residuals)
adf_result = adfuller(residuals)
print ("p-val of the ADF test for residuals : ", adf_result[1])
additive = seasonal.seasonal_decompose(ibm['Close'], freq=1000, model='additive')
def plot_decomposed(decompose_result):
    fig, ax = plt.subplots(4, sharex=True)
    fig.set_size_inches(15, 10)
    
    ibm['Close'].plot(ax=ax[0], color='b', linestyle='-')
    ax[0].set_title('IBM Close price')
    
    pd.Series(data=decompose_result.trend, index=ibm.index).plot(ax=ax[1], color='r', linestyle='-')
    ax[1].set_title('Trend line')
    
    pd.Series(data=decompose_result.seasonal, index=ibm.index).plot(ax=ax[2], color='g', linestyle='-')
    ax[2].set_title('Seasonal component')
    
    pd.Series(data=decompose_result.resid, index=ibm.index).plot(ax=ax[3], color='k', linestyle='-')
    ax[3].set_title('Irregular variables')
plot_decomposed(additive)
adf_result = adfuller(additive.resid[np.where(np.isfinite(additive.resid))[0]])
print("p-value for the irregular variations of the additive model : ", adf_result[1])
multiplicative = seasonal.seasonal_decompose(ibm['Close'], freq=1000, model='multiplicative')
plot_decomposed(multiplicative)
adf_result_multiplicative = adfuller(multiplicative.resid[np.where(np.isfinite(multiplicative.resid))[0]])
print("p-value for the irregular variations of the additive model : ", adf_result_multiplicative[1])