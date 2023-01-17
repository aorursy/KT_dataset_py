import os

import sys



import pandas as pd

import pandas_datareader.data as web

import numpy as np



import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline

p = print

MSFT = pd.read_csv('../input/Microsoft.csv', header=None)

AMZN =  pd.read_csv('../input/Amazon.csv',header=None)

MSFT['AdjClose'] =  pd.read_csv('../input/Microsoft.csv', header=None)

AMZN['AdjClose'] =  pd.read_csv('../input/Amazon.csv', header=None)
def tsplot(y, lags=None, figsize=(15, 15), style='bmh'):

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        layout = (3, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        

        y.plot(ax=ts_ax)

        ts_ax.set_title('Time Series Analysis Plots')

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.7)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.7)

        plt.tight_layout()

    return 
np.random.seed(1)



# plot of discrete white noise

whnoise = np.random.normal(size=1000)

tsplot(whnoise, lags=30)
p("Outputs\n-------------\nMean: {:.3f}\nVariance: {:.3f}\nStandard Deviation: {:.3f}"

.format(whnoise.mean(), whnoise.var(), whnoise.std()))
np.random.seed(1)

n_samples = 1000



x = w = np.random.normal(size=n_samples)

for t in range(n_samples):

    x[t] = x[t-1] + w[t]



tsplot(x, lags=30)
tsplot(np.diff(x), lags=30)
tsplot(MSFT.AdjClose, lags=30)
tsplot(np.diff(MSFT.AdjClose), lags=30)
tsplot(AMZN.AdjClose, lags=30)
tsplot(np.diff(AMZN.AdjClose), lags=30)