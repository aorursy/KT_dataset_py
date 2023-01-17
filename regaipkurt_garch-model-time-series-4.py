!pip3 install arch
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as sgt

import statsmodels.tsa.stattools as sts

from statsmodels.tsa.arima_model import ARIMA

from scipy.stats.distributions import chi2 

from arch import arch_model

from math import sqrt

import seaborn as sns

sns.set()
raw_csv_data = pd.read_csv("../input/financial-markets/Index2018.csv") 

df_comp=raw_csv_data.copy()

df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)

df_comp.set_index("date", inplace=True)

df_comp=df_comp.asfreq('b')

df_comp=df_comp.fillna(method='ffill')
df_comp['market_value']=df_comp.ftse
del df_comp['spx']

del df_comp['dax']

del df_comp['ftse']

del df_comp['nikkei']

size = int(len(df_comp)*0.8)

df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]
import warnings

warnings.filterwarnings("ignore")
def LLR_test(mod_1, mod_2, DF = 1):

    L1 = mod_1.fit(start_ar_lags = 11).llf

    L2 = mod_2.fit(start_ar_lags = 11).llf

    LR = (2*(L2-L1))    

    p = chi2.sf(LR, DF).round(3)

    return p
df['returns'] = df.market_value.pct_change(1)*100
model_garch_1_1 = arch_model(df.returns[1:], mean = "Constant", vol = "GARCH", p = 1, q = 1)

results_garch_1_1 = model_garch_1_1.fit(update_freq = 5)

results_garch_1_1.summary()
model_garch_1_2 = arch_model(df.returns[1:], mean = "Constant",  vol = "GARCH", p = 1, q = 2)

results_garch_1_2 = model_garch_1_2.fit(update_freq = 5)

results_garch_1_2.summary()
model_garch_1_3 = arch_model(df.returns[1:], mean = "Constant",  vol = "GARCH", p = 1, q = 3)

results_garch_1_3 = model_garch_1_3.fit(update_freq = 5)

results_garch_1_3.summary()
model_garch_2_1 = arch_model(df.returns[1:], mean = "Constant",  vol = "GARCH", p = 2, q = 1)

results_garch_2_1 = model_garch_2_1.fit(update_freq = 5)

results_garch_2_1.summary()
model_garch_3_1 = arch_model(df.returns[1:], mean = "Constant",  vol = "GARCH", p = 3, q = 1)

results_garch_3_1 = model_garch_3_1.fit(update_freq = 5)

results_garch_3_1.summary()