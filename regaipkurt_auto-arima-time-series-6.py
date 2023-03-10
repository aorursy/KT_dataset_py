!pip3 install arch yfinance pmdarima
import numpy as np

import pandas as pd

import scipy

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from statsmodels.tsa.arima_model import ARIMA

from arch import arch_model

import seaborn as sns

import yfinance

import warnings

warnings.filterwarnings("ignore")

sns.set()
raw_data = yfinance.download (tickers = "^GSPC ^FTSE ^N225 ^GDAXI", start = "1994-01-07", end = "2020-06-30", 

                              interval = "1d", group_by = 'ticker', auto_adjust = True, treads = True)
df_comp = raw_data.copy()
df_comp['spx'] = df_comp['^GSPC'].Close[:]

df_comp['dax'] = df_comp['^GDAXI'].Close[:]

df_comp['ftse'] = df_comp['^FTSE'].Close[:]

df_comp['nikkei'] = df_comp['^N225'].Close[:]
df_comp = df_comp.iloc[1:]

del df_comp['^N225']

del df_comp['^GSPC']

del df_comp['^GDAXI']

del df_comp['^FTSE']

df_comp=df_comp.asfreq('b')

df_comp=df_comp.fillna(method='ffill')
df_comp['ret_spx'] = df_comp.spx.pct_change(1)*100

df_comp['ret_ftse'] = df_comp.ftse.pct_change(1)*100

df_comp['ret_dax'] = df_comp.dax.pct_change(1)*100

df_comp['ret_nikkei'] = df_comp.nikkei.pct_change(1)*100
size = int(len(df_comp)*0.8)

df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]
from pmdarima.arima import auto_arima
model_auto = auto_arima(df.ret_ftse[1:])
model_auto
model_auto.summary()
model_auto = auto_arima(df_comp.ret_ftse[1:], exogenous = df_comp[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], m = 5,

                       max_order = None, max_p = 7, max_q = 7, max_d = 2, max_P = 4, max_Q = 4, max_D = 2,

                       maxiter = 50, alpha = 0.05, n_jobs = -1, trend = 'ct', information_criterion = 'oob',

                       out_of_sample = int(len(df_comp)*0.2))



# exogenous -> outside factors (e.g other time series)

# m -> seasonal cycle length

# max_order -> maximum amount of variables to be used in the regression (p + q)

# max_p -> maximum AR components

# max_q -> maximum MA components

# max_d -> maximum Integrations

# maxiter -> maximum iterations we're giving the model to converge the coefficients (becomes harder as the order increases)

# alpha -> level of significance, default is 5%, which we should be using most of the time

# n_jobs -> how many models to fit at a time (-1 indicates "as many as possible")

# trend -> "ct" usually

# information_criterion -> 'aic', 'aicc', 'bic', 'hqic', 'oob' 

#        (Akaike Information Criterion, Corrected Akaike Information Criterion,

#        Bayesian Information Criterion, Hannan-Quinn Information Criterion, or

#        "out of bag"--for validation scoring--respectively)

# out_of_smaple -> validates the model selection (pass the entire dataset, and set 20% to be the out_of_sample_size)
model_auto.summary()
plt.figure(figsize=(15,7))

plt.plot(model_auto.resid())

plt.title("ARIMA MODEL ERRORS", size=25)

plt.show()