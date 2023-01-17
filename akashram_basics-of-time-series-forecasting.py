import numpy as np 

import pandas as pd 

import os

from fbprophet import Prophet

from matplotlib import pyplot as plt

import plotly.express as px

from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
itc = pd.read_csv("/kaggle/input/nifty50-stock-market-data/ITC.csv")

itc.set_index("Date", drop=False, inplace=True)

itc.head()
print(f'There are {itc.shape[0]} rows and {itc.shape[1]} columns')
itcv = itc.VWAP.reset_index()

fig = px.line(itcv, x="Date", y="VWAP", title='VMAP of ITC over the years')

fig.show()
df_vwap = itc[['Date','VWAP']]

df_vwap['Date'] = df_vwap['Date'].apply(pd.to_datetime)

df_vwap.set_index("Date", inplace = True)

df_vwap.head()
vwap_2005 = df_vwap['2005':'2006']

vwap_2005 = vwap_2005.reset_index()

fig = px.line(vwap_2005, x="Date", y="VWAP", title='VMAP of ITC over the years')

fig.show()
itcv = itc.VWAP.reset_index()

fig = px.line(itcv, x="Date", y="VWAP", title='VMAP of ITC over the years')

fig.show()
vwap_itc_2020 = df_vwap['2019-04':'2020-05']

vwap_itc_2020['mavg_12'] = vwap_itc_2020['VWAP'].rolling(window = 12).mean().shift(1)

pd.set_option('display.float_format', lambda x: '%.2f'% x)

mayd = vwap_itc_2020.tail(19)

mayd = mayd.reset_index()
import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter(

    x=mayd['Date'],

    y=mayd['VWAP'],

    name="Actual VWAP"

))



fig.add_trace(go.Scatter(

    x=mayd['Date'],

    y=mayd['mavg_12'],

    name="Moving Average"

))



fig.update_layout(

    title="VWAP - Actual vs Moving Average of ITC for the Month of May 2020",

    xaxis_title="Date",

    yaxis_title="VWAP",

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="#7f7f7f"

    )

)



fig.show()
def get_mape(actual, predicted):

    y_true, y_pred = np.array(actual), np.array(predicted)

    return np.round( np.mean(np.abs((actual - predicted)/ actual)) * 100, 2)
get_mape(mayd['VWAP'].values, mayd['mavg_12'].values)
from sklearn.metrics import mean_squared_error 

np.sqrt(mean_squared_error(mayd['VWAP'].values, mayd['mavg_12'].values))
vwap_itc_2020['ewm'] = vwap_itc_2020['VWAP'].ewm(alpha = 0.2 ).mean()

pd.set_option('display.float_format', lambda x: '%.2f'% x)

emayd = vwap_itc_2020.tail(19)

emayd = emayd.reset_index()
import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['VWAP'],

    name="Actual VWAP"

))



fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['ewm'],

    name="Exponential Smoothing"

))



fig.update_layout(

    title="VWAP - Actual vs Exponential Smoothing of ITC for the Month of May 2020",

    xaxis_title="Date",

    yaxis_title="VWAP",

    font=dict(

        family="Courier New, monospace",

        size=8,

        color="#7f7f7f"

    )

)



fig.show()
import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['VWAP'],

    name="Actual VWAP"

))





fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['mavg_12'],

    name="Moving Average"

))



fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['ewm'],

    name="Exponential Smoothing"

))



fig.update_layout(

    title="VWAP - Actual vs vs Moving Average vs Exponential Smoothing of ITC for the Month of May 2020",

    xaxis_title="Date",

    yaxis_title="VWAP",

    font=dict(

        family="Courier New, monospace",

        size=8,

        color="#7f7f7f"

    )

)



fig.show()
vwap_itc_20201 = vwap_itc_2020.reset_index()

from statsmodels.tsa.seasonal import seasonal_decompose



# Multiplicative Decomposition 

result_mul = seasonal_decompose(vwap_itc_2020['VWAP'], model='multiplicative', extrapolate_trend = 'freq', freq=12)



# Additive Decomposition

result_add = seasonal_decompose(vwap_itc_2020['VWAP'], model='additive', extrapolate_trend = 'freq', freq=12)



# Plot

plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=15)
result_add.plot().suptitle('Additive Decompose', fontsize=15)
vwap_itc_2020['mul_seasonal'] = result_mul.seasonal

vwap_itc_2020['mul_trend'] = result_mul.trend



vwap_itc_2020['add_seasonal'] = result_add.seasonal

vwap_itc_2020['add_trend'] = result_add.trend
import plotly.graph_objects as go

fig = go.Figure()



vwap_itc_2020d = vwap_itc_2020.reset_index()



fig.add_trace(go.Scatter(

    x=vwap_itc_2020d['Date'],

    y=vwap_itc_2020d['mul_seasonal'],

    name="Multiplicative Seasonal Component"

))



fig.add_trace(go.Scatter(

    x=vwap_itc_2020d['Date'],

    y=vwap_itc_2020d['add_seasonal'],

    name="Additive Seasonal Component"

))



fig.update_layout(

    title="Seasonal Component",

    xaxis_title="Date",

    yaxis_title="VWAP",

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="#7f7f7f"

    )

)



fig.show()
import plotly.graph_objects as go

fig = go.Figure()



vwap_itc_2020d = vwap_itc_2020.reset_index()



fig.add_trace(go.Scatter(

    x=vwap_itc_2020d['Date'],

    y=vwap_itc_2020d['mul_trend'],

    name="Multiplicative Trend"

))



fig.add_trace(go.Scatter(

    x=vwap_itc_2020d['Date'],

    y=vwap_itc_2020d['add_trend'],

    name="Additive Trend"

))



fig.update_layout(

    title="Trend Component",

    xaxis_title="Date",

    yaxis_title="VWAP",

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="#7f7f7f"

    )

)



fig.show()
#ACF plot to show auto-correlation upto lag of 20

    

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({'figure.figsize': (10,6)})

acf_plot = plot_acf(vwap_itc_2020d.VWAP, lags=20)
plt.rcParams.update({'figure.figsize': (10,6)})



acf_plot = plot_pacf(vwap_itc_2020d.VWAP, lags=20)
from statsmodels.tsa.arima_model import ARIMA

arima = ARIMA(vwap_itc_2020.VWAP[0:265].astype(np.float64), order = (1,0,0))

arima1 = ARIMA(vwap_itc_2020.VWAP[0:265].astype(np.float64), order = (0,0,0))

ar_model = arima.fit()

ar_model1 = arima1.fit()

ar_model.summary2()
forecast_arima = ar_model.predict(265, 283)

forecast_arima1 = ar_model1.predict(265, 283)



get_mape(vwap_itc_2020.VWAP[265:285].values, forecast_arima.values)
get_mape(vwap_itc_2020.VWAP[265:285].values, forecast_arima1.values)
arima = ARIMA(vwap_itc_2020.VWAP[0:265].astype(np.float64), order = (0,0,1))

ma_model = arima.fit()

ma_model.summary2()
forecast_arima = ma_model.predict(265, 283)



get_mape(vwap_itc_2020.VWAP[265:285].values, forecast_arima.values)
arima = ARIMA(vwap_itc_2020.VWAP[0:265].astype(np.float64), order = (1,0,1)) 

arma_model = arima.fit() 

arma_model.summary2()
forecast_arma = arma_model.predict(265, 283)

get_mape(vwap_itc_2020.VWAP[265:285].values, forecast_arma.values)
from statsmodels.tsa.stattools import adfuller



def adfuller_test(ts): 

    adfuller_result = adfuller(ts, autolag=None)

    adfuller_out = pd.Series(adfuller_result[0:4],

                             index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

    print(adfuller_out)
adfuller_test(vwap_itc_2020.VWAP)
vwap_itc_2020['vdiff'] = vwap_itc_2020.VWAP - vwap_itc_2020.VWAP.shift(1)

vwap_itc_2020.head(5)

vwap_itc_2020 = vwap_itc_2020.dropna()
import plotly.graph_objects as go

fig = go.Figure()



vwap_itc_20201 = vwap_itc_2020.reset_index()



fig.add_trace(go.Scatter(

    x=vwap_itc_20201['Date'],

    y=vwap_itc_20201['vdiff'],

    name="First-Order Differencing"

))



fig.update_layout(

    xaxis_title="Date",

    yaxis_title="First-order difference",

    font=dict(

        family="Courier New, monospace",

        size=12,

        color="#7f7f7f"

    )

)



fig.show()
pacf_plot = plot_acf(vwap_itc_20201.vdiff.dropna(), lags=10)
arima = ARIMA(vwap_itc_2020.VWAP[0:265].astype(np.float64), order = (1,1,1)) 

arima_model = arima.fit()

arima_model.summary2()
predict, stderr, ci = arima_model.forecast(steps=7)

get_mape(vwap_itc_2020.VWAP[265:285].values, predict)