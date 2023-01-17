import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.graphics.tsaplots as sgt

import statsmodels.tsa.stattools as sts

from statsmodels.tsa.arima_model import ARMA

from scipy.stats.distributions import chi2 

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

    L1 = mod_1.fit().llf

    L2 = mod_2.fit().llf

    LR = (2*(L2-L1))    

    p = chi2.sf(LR, DF).round(3)

    return p
df['returns'] = df.market_value.pct_change(1)*100
model_ret_ar_1_ma_1 = ARMA(df.returns[1:], order=(1,1))

results_ret_ar_1_ma_1 = model_ret_ar_1_ma_1.fit()

results_ret_ar_1_ma_1.summary()
model_ret_ar_1 = ARMA(df.returns[1:], order = (1,0))

model_ret_ma_1 = ARMA(df.returns[1:], order = (0,1))
print("\nARMA vs AR ", LLR_test(model_ret_ar_1, model_ret_ar_1_ma_1))

print("\nARMA vs MA ", LLR_test(model_ret_ma_1, model_ret_ar_1_ma_1))
model_ret_ar_3_ma_3 = ARMA(df.returns[1:], order=(3,3))

results_ret_ar_3_ma_3 = model_ret_ar_3_ma_3.fit()
LLR_test(model_ret_ar_1_ma_1, model_ret_ar_3_ma_3, DF = 4)
results_ret_ar_3_ma_3.summary()
model_ret_ar_3_ma_2 = ARMA(df.returns[1:], order=(3,2))

results_ret_ar_3_ma_2 = model_ret_ar_3_ma_2.fit()

results_ret_ar_3_ma_2.summary()
model_ret_ar_2_ma_3 = ARMA(df.returns[1:], order=(2,3))

results_ret_ar_2_ma_3 = model_ret_ar_2_ma_3.fit()

results_ret_ar_2_ma_3.summary()
LLR_test(model_ret_ar_2_ma_3, model_ret_ar_3_ma_3)
model_ret_ar_3_ma_1 = ARMA(df.returns[1:], order=(3,1))

results_ret_ar_3_ma_1 = model_ret_ar_3_ma_1.fit()

results_ret_ar_3_ma_1.summary()
LLR_test(model_ret_ar_3_ma_1, model_ret_ar_3_ma_2)
model_ret_ar_2_ma_2 = ARMA(df.returns[1:], order=(2,2))

results_ret_ar_2_ma_2 = model_ret_ar_2_ma_2.fit()

results_ret_ar_2_ma_2.summary()
model_ret_ar_1_ma_3 = ARMA(df.returns[1:], order=(1,3))

results_ret_ar_1_ma_3 = model_ret_ar_1_ma_3.fit()

results_ret_ar_1_ma_3.summary()
print("\n ARMA(3,2): \tLL = ", results_ret_ar_3_ma_2.llf, "\tAIC = ", results_ret_ar_3_ma_2.aic)

print("\n ARMA(1,3): \tLL = ", results_ret_ar_1_ma_3.llf, "\tAIC = ", results_ret_ar_1_ma_3.aic)
df['res_ret_ar_3_ma_2'] = results_ret_ar_3_ma_2.resid[1:]
df.res_ret_ar_3_ma_2.plot(figsize = (20,5))

plt.title("Residuals of Returns", size=24)

plt.show()
sgt.plot_acf(df.res_ret_ar_3_ma_2[2:], zero = False, lags = 40)

plt.title("ACF Of Residuals for Returns",size=24)

plt.show()
model_ret_ar_5_ma_5 = ARMA(df.returns[1:], order=(5,5))

results_ret_ar_5_ma_5 = model_ret_ar_5_ma_5.fit()

results_ret_ar_5_ma_5.summary()
model_ret_ar_5_ma_1 = ARMA(df.returns[1:], order=(5,1))

results_ret_ar_5_ma_1 = model_ret_ar_5_ma_1.fit()

results_ret_ar_5_ma_1.summary()
model_ret_ar_1_ma_5 = ARMA(df.returns[1:], order=(1,5))

results_ret_ar_1_ma_5 = model_ret_ar_1_ma_5.fit()

results_ret_ar_1_ma_5.summary()
print("ARMA(5,1):  \t LL = ",results_ret_ar_5_ma_1.llf,"\t AIC = ",results_ret_ar_5_ma_1.aic)

print("ARMA(1,5):  \t LL = ",results_ret_ar_1_ma_5.llf,"\t AIC = ",results_ret_ar_1_ma_5.aic)
print("ARMA(3,2):  \t LL = ",results_ret_ar_3_ma_2.llf,"\t AIC = ",results_ret_ar_3_ma_2.aic)
df['res_ret_ar_5_ma_1'] = results_ret_ar_5_ma_1.resid
sgt.plot_acf(df.res_ret_ar_5_ma_1[1:], zero = False, lags = 40)

plt.title("ACF of Residuals for Returns",size=24)

plt.show()
sgt.plot_acf(df.market_value, unbiased=True, zero = False, lags = 40)

plt.title("Autocorrelation Function for Prices",size=20)

plt.show()
sgt.plot_pacf(df.market_value, lags = 40, alpha = 0.05, zero = False , method = ('ols'))

plt.title("Partial Autocorrelation Function for Prices",size=20)

plt.show()
model_ar_1_ma_1 = ARMA(df.market_value, order=(1,1))

results_ar_1_ma_1 = model_ar_1_ma_1.fit()

results_ar_1_ma_1.summary()
df['res_ar_1_ma_1'] = results_ar_1_ma_1.resid
sgt.plot_acf(df.res_ar_1_ma_1, zero = False, lags = 40)

plt.title("ACF Of Residuals of Prices",size=20)

plt.show()
model_ar_6_ma_6 = ARMA(df.market_value, order=(6,6))

results_ar_6_ma_6 = model_ar_6_ma_6.fit(start_ar_lags = 11)

results_ar_6_ma_6.summary()
model_ar_5_ma_6 = ARMA(df.market_value, order=(5,6))

results_ar_5_ma_6 = model_ar_5_ma_6.fit(start_ar_lags = 7)

results_ar_5_ma_6.summary()
model_ar_6_ma_1 = ARMA(df.market_value, order=(6,1))

results_ar_6_ma_1 = model_ar_6_ma_1.fit(start_ar_lags = 7)

results_ar_6_ma_1.summary()
print("ARMA(5,6):  \t LL = ", results_ar_5_ma_6.llf, "\t AIC = ", results_ar_5_ma_6.aic)

print("ARMA(6,1):  \t LL = ", results_ar_6_ma_1.llf, "\t AIC = ", results_ar_6_ma_1.aic)
df['res_ar_5_ma_6'] = results_ar_5_ma_6.resid

sgt.plot_acf(df.res_ar_5_ma_6, zero = False, lags = 40)

plt.title("ACF Of Residuals of Prices",size=20)

plt.show()
print("ARMA(5,6):  \t LL = ", results_ar_5_ma_6.llf, "\t AIC = ", results_ar_5_ma_6.aic)

print("ARMA(5,1):  \t LL = ", results_ret_ar_5_ma_1.llf, "\t AIC = ", results_ret_ar_5_ma_1.aic)