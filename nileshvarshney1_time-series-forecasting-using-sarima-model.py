# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        full_filepath = os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read AirPassangers data

ts = pd.read_csv(os.path.join(full_filepath),parse_dates=["Month"], index_col="Month")

display(ts.shape)

print(ts.head())

print('Timeseries Range => ', ts.index.min(), ' - ' , ts.index.max())
def draw_ts_plot(timeseries, xlabel ='Date', ylabel ='Value', title ="", dpi=120):

    plt.figure(figsize=(16,5),dpi=dpi)

    plt.plot(timeseries, color='red')

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()



draw_ts_plot(ts, xlabel="Dates in Month", ylabel = "Passanger Counts", title="Monthly Airlines passange Counts")
years = ts.index.year.unique()

plt.figure(figsize=(16,5),dpi=120)

for year in years:

    plt.plot(ts.index[ts.index.year == year].month,

    ts[ts.index.year == year]['#Passengers'], label = year )



plt.gca().set(title = "Yearly Trend")

plt.legend(loc='right')

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose



# additive decomposition

result_additive = seasonal_decompose(ts,model='additive', extrapolate_trend='freq')



# multiplicative

result_multiplicative = seasonal_decompose(ts,model='multiplicative', extrapolate_trend='freq')



# plot

plt.rcParams.update({'figure.figsize':(20,8)})

result_additive.plot()

plt.suptitle('Additive Seasonal Decompose', fontsize=12)

plt.show()



result_multiplicative.plot()

plt.suptitle('Multiplicative Seasonal Decompose', fontsize=12)

plt.show()
df_multiplicative = pd.concat([

    result_multiplicative.observed, 

    result_multiplicative.trend, 

    result_multiplicative.seasonal, 

    result_multiplicative.resid], axis= 1)

df_multiplicative.columns  = ['actual', 'trend','seasonal','resid']

df_multiplicative.head()
train = ts[0:-36]

test = ts[-36:]

print('Train Timeseries Range => ', train.index.min(), ' - ' , train.index.max())

print('Train Timeseries Range => ', test.index.min(), ' - ' , test.index.max())
# regression{“c”,”ct”,”ctt”,”nc”}

# Constant and trend order to include in regression.



# “c” : constant only (default).

# “ct” : constant and trend.

# “ctt” : constant, and linear and quadratic trend.

# “nc” : no constant, no trend.



for reg in ["c","ct","ctt","nc"]:

    res = sm.tsa.adfuller(train.dropna(),regression=reg)

    print('Reg - {}\t adf :{} - lag used : {}, Critical value : {}'.format(reg, res[0],res[2],res[4]))

    res = sm.tsa.adfuller(train.diff().dropna(),regression=reg)

    print('Reg diff - {}\t adf :{} - lagused : {}, Critical value : {}'.format(reg, res[0],res[2],res[4]))
fig, ax = plt.subplots(2,1, figsize =(18, 10))

fig = sm.graphics.tsa.plot_acf(train.diff().dropna(), lags=15, ax=ax[0])

fig = sm.graphics.tsa.plot_pacf(train.diff().dropna(), lags=15, ax=ax[1])

plt.show()
import warnings

warnings.filterwarnings('ignore')

res = sm.tsa.arma_order_select_ic(train, max_ar=7, max_ma=7, ic=['aic'], trend='nc')

print(res['aic_min_order'])
arima = sm.tsa.statespace.SARIMAX(train,order=(6,1,7), seasonal_order=(2,1,0,12),

                                 enforce_stationarity=False, enforce_invertibility=False,).fit()

#arima.summary()

from sklearn.metrics import mean_squared_error

pred_train = arima.predict(train.index.min(), train.index.max())

pred_test = arima.predict(test.index.min(), test.index.max())

plt.title('ARIMA model MSE:{}'.format(mean_squared_error(test,pred_test)))

plt.plot(train, label='train')

plt.plot(pred_train, color='orange', linestyle='--', label= 'train prediction')

plt.plot(pred_test, color='red', linestyle='--', label= 'prediction')

plt.plot(test, color='green', label='actual')

plt.legend(loc='best')

plt.show()