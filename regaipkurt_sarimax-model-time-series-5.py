import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as sgt

import statsmodels.tsa.stattools as sts

from statsmodels.tsa.arima_model import ARIMA

from scipy.stats.distributions import chi2 

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
import warnings

warnings.filterwarnings("ignore")
#del df_comp['spx']

#del df_comp['dax']

#del df_comp['ftse']

#del df_comp['nikkei']

size = int(len(df_comp)*0.8)

df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]
def LLR_test(mod_1, mod_2, DF = 1):

    L1 = mod_1.llf

    L2 = mod_2.llf

    LR = (2*(L2-L1))    

    p = chi2.sf(LR, DF).round(3)

    return p
df['returns'] = df.market_value.pct_change(1)*100
model_ar_1_i_1_ma_1 = ARIMA(df.market_value, order=(1,1,1))

results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()

results_ar_1_i_1_ma_1.summary()
df['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid

sgt.plot_acf(df.res_ar_1_i_1_ma_1, zero = False, lags = 40)

plt.title("ACF Of Residuals for ARIMA(1,1,1)",size=20)

plt.show()
df['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid.iloc[:]

sgt.plot_acf(df.res_ar_1_i_1_ma_1[1:], zero = False, lags = 40)

plt.title("ACF Of Residuals for ARIMA(1,1,1)",size=20)

plt.show()
model_ar_1_i_1_ma_2 = ARIMA(df.market_value, order=(1,1,2))

results_ar_1_i_1_ma_2 = model_ar_1_i_1_ma_2.fit()

model_ar_1_i_1_ma_3 = ARIMA(df.market_value, order=(1,1,3))

results_ar_1_i_1_ma_3 = model_ar_1_i_1_ma_3.fit()

model_ar_2_i_1_ma_1 = ARIMA(df.market_value, order=(2,1,1))

results_ar_2_i_1_ma_1 = model_ar_2_i_1_ma_1.fit()

model_ar_3_i_1_ma_1 = ARIMA(df.market_value, order=(3,1,1))

results_ar_3_i_1_ma_1 = model_ar_3_i_1_ma_1.fit()

model_ar_3_i_1_ma_2 = ARIMA(df.market_value, order=(3,1,2))

results_ar_3_i_1_ma_2 = model_ar_3_i_1_ma_2.fit(start_ar_lags=5)
print("ARIMA(1,1,1):  \t LL = ", results_ar_1_i_1_ma_1.llf, "\t AIC = ", results_ar_1_i_1_ma_1.aic)

print("ARIMA(1,1,2):  \t LL = ", results_ar_1_i_1_ma_2.llf, "\t AIC = ", results_ar_1_i_1_ma_2.aic)

print("ARIMA(1,1,3):  \t LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)

print("ARIMA(2,1,1):  \t LL = ", results_ar_2_i_1_ma_1.llf, "\t AIC = ", results_ar_2_i_1_ma_1.aic)

print("ARIMA(3,1,1):  \t LL = ", results_ar_3_i_1_ma_1.llf, "\t AIC = ", results_ar_3_i_1_ma_1.aic)

print("ARIMA(3,1,2):  \t LL = ", results_ar_3_i_1_ma_2.llf, "\t AIC = ", results_ar_3_i_1_ma_2.aic)
print("\nLLR test p-value = " + str(LLR_test(results_ar_1_i_1_ma_2, results_ar_1_i_1_ma_3)))
print("\nLLR test p-value = " + str(LLR_test(results_ar_1_i_1_ma_1, results_ar_1_i_1_ma_3, DF = 2)))
df['res_ar_1_i_1_ma_3'] = results_ar_1_i_1_ma_3.resid

sgt.plot_acf(df.res_ar_1_i_1_ma_3[1:], zero = False, lags = 40)

plt.title("ACF Of Residuals for ARIMA(1,1,3)", size=20)

plt.show()
model_ar_5_i_1_ma_1 = ARIMA(df.market_value, order=(5,1,1))

results_ar_5_i_1_ma_1 = model_ar_5_i_1_ma_1.fit(start_ar_lags=11)

model_ar_6_i_1_ma_3 = ARIMA(df.market_value, order=(6,1,3))

results_ar_6_i_1_ma_3 = model_ar_6_i_1_ma_3.fit(start_ar_lags=11)
results_ar_5_i_1_ma_1.summary()
print("ARIMA(1,1,3):  \t LL = ", results_ar_1_i_1_ma_3.llf, "\t AIC = ", results_ar_1_i_1_ma_3.aic)

print("ARIMA(5,1,1):  \t LL = ", results_ar_5_i_1_ma_1.llf, "\t AIC = ", results_ar_5_i_1_ma_1.aic)

print("ARIMA(6,1,3):  \t LL = ", results_ar_6_i_1_ma_3.llf, "\t AIC = ", results_ar_6_i_1_ma_3.aic)
print("\nLLR test p-value = " + str(LLR_test(results_ar_1_i_1_ma_3, results_ar_6_i_1_ma_3, DF = 5)))
print("\nLLR test p-value = " + str(LLR_test(results_ar_5_i_1_ma_1, results_ar_6_i_1_ma_3, DF = 3)))
df['res_ar_5_i_1_ma_1'] = results_ar_5_i_1_ma_1.resid

sgt.plot_acf(df.res_ar_5_i_1_ma_1[1:], zero = False, lags = 40)

plt.title("ACF Of Residuals for ARIMA(5,1,1)", size=20)

plt.show()
df['delta_prices']=df.market_value.diff(1)
model_delta_ar_1_i_1_ma_1 = ARIMA(df.delta_prices[1:], order=(1,0,1))

results_delta_ar_1_i_1_ma_1 = model_delta_ar_1_i_1_ma_1.fit()

results_delta_ar_1_i_1_ma_1.summary()
sts.adfuller(df.delta_prices[1:])
model_ar_1_i_2_ma_1 = ARIMA(df.market_value, order=(1,2,1))

results_ar_1_i_2_ma_1 = model_ar_1_i_2_ma_1.fit(start_ar_lags=10)

results_ar_1_i_2_ma_1.summary()
df['res_ar_1_i_2_ma_1'] = results_ar_1_i_2_ma_1.resid.iloc[:]

sgt.plot_acf(df.res_ar_1_i_2_ma_1[2:], zero = False, lags = 40)

plt.title("ACF Of Residuals for ARIMA(1,2,1)",size=20)

plt.show()
model_ar_1_i_1_ma_1_Xspx = ARIMA(df.market_value, exog = df.spx, order=(1,1,1))

results_ar_1_i_1_ma_1_Xspx = model_ar_1_i_1_ma_1_Xspx.fit()

results_ar_1_i_1_ma_1_Xspx.summary()
from statsmodels.tsa.statespace.sarimax import SARIMAX
model_sarimax = SARIMAX(df.market_value, exog = df.spx, order=(1,0,1), seasonal_order = (2,0,1,5))

results_sarimax = model_sarimax.fit()

results_sarimax.summary()