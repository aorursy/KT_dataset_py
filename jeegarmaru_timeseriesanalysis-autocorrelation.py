import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def get_stock_data(symbol):
    df = pd.read_csv("../input/Data/Stocks/{}.us.txt".format(symbol), index_col='Date', parse_dates=True, 
                     na_values='nan', usecols=['Date', 'Close'])
    return df
ibm = get_stock_data('ibm')

ibm.plot(figsize=(15, 5))
lag_plot(ibm['Close'])
plt.clf()
fig, ax = plt.subplots(figsize=(15, 5))
autocorrelation_plot(ibm['Close'], ax=ax)
plt.clf()
fig, ax = plt.subplots(figsize=(15, 10))
plot_acf(ibm['Close'], lags=1200, use_vlines=False, ax=ax)
fig, ax = plt.subplots(figsize=(15, 10))
plot_pacf(ibm['Close'], lags=100, use_vlines=False, ax=ax)
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
calculate_acf(ibm['Close'])
adf, p_value, usedlag, nobs, critical_values, *values = adfuller(ibm['Close'])
print ("ADF is ", adf)
print ("p value is ", p_value)
print ("lags used are ", usedlag)
print ("Number of observations are ", nobs)
print ("Critical Values are", critical_values)
ddiff = ibm.diff(1)[1:]
ddiff.plot()
fig.clf()
fig, ax = plt.subplots(figsize=(10, 10))
lag_plot(ddiff['Close'], ax=ax)
fig.clf()
fig, ax = plt.subplots(figsize=(15, 8))
autocorrelation_plot(ddiff, ax=ax)
calculate_acf(ddiff['Close'])