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



!pip install bubbly
df = pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv")



print(f'There are {df.shape[0]} rows and {df.shape[1]} columns')
df_reg = df.groupby(['Region', 'Year'])['AvgTemperature'].mean().reset_index()

df_reg = df_reg[(df_reg['AvgTemperature']> 0 )]
import plotly.express as px

fig = px.scatter(df_reg, x="Year", y="AvgTemperature", color="Region",

                 size='AvgTemperature', hover_data=['Region'])



fig.show()
from bubbly.bubbly import bubbleplot 

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



figure = bubbleplot(dataset=df_reg, x_column='Year', y_column='AvgTemperature', 

    bubble_column='Region', time_column='Year', color_column='Region', 

    x_title="Year", y_title="Avg Temperature", title='Avg Temperature over the years across Continents',

    scale_bubble=3, height=650)



iplot(figure, config={'scrollzoom': True})
df_delhi = df[df['City'] == 'Delhi']

df_delhi['Date'] = pd.to_datetime(df_delhi[['Year', 'Month', 'Day']])

#df_months = df_delhi.groupby(['Year', 'Month', 'Day'])['AvgTemperature'].mean().reset_index()

df_months = df_delhi[df_delhi['Year']>2017]



df_months = df_months.reset_index()

df_months.drop('index', axis=1, inplace=True)
df_months = df_months[['Date', 'AvgTemperature']]

df_months['Date'] = df_months['Date'].apply(pd.to_datetime)

df_months.set_index("Date", inplace = True)

df_months = df_months['2018-01':'2020-04']
df_months_ = df_months.reset_index()

fig = px.line(df_months_, x="Date", y="AvgTemperature", title='Average Temperature of Delhi over the last 2 years')

fig.show()
df_months['mavg_12'] = df_months['AvgTemperature'].rolling(window = 12).mean().shift(1)

pd.set_option('display.float_format', lambda x: '%.2f'% x)

mayd = df_months.reset_index()

mayd = mayd.tail(30)
import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter(

    x=mayd['Date'],

    y=mayd['AvgTemperature'],

    name="Actual Avg Temperature"

))



fig.add_trace(go.Scatter(

    x=mayd['Date'],

    y=mayd['mavg_12'],

    name="Moving Average"

))



fig.update_layout(

    title="Avg Temp. - Actual vs Moving Average of Delhi for the Month of April 2020",

    xaxis_title="Date",

    yaxis_title="Avg Temp",

    font=dict(

        family="Courier New, monospace",

        size=9.5,

        color="#7f7f7f"

    )

)



fig.show()
def get_mape(actual, predicted):

    y_true, y_pred = np.array(actual), np.array(predicted)

    return np.round( np.mean(np.abs((actual - predicted)/ actual)) * 100, 2)
get_mape(mayd['AvgTemperature'].values, mayd['mavg_12'].values)
from sklearn.metrics import mean_squared_error 

np.sqrt(mean_squared_error(mayd['AvgTemperature'].values, mayd['mavg_12'].values))
df_months['ewm'] = df_months['AvgTemperature'].ewm(alpha = 0.2 ).mean()

pd.set_option('display.float_format', lambda x: '%.2f'% x)

emayd = df_months.tail(30)

emayd = emayd.reset_index()
import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['AvgTemperature'],

    name="Actual AvgTemperature"

))



fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['ewm'],

    name="Exponential Smoothing"

))



fig.update_layout(

    title="AvgTemperature - Actual vs Exponential Smoothing of Delhi for the Month of Apr 2020",

    xaxis_title="Date",

    yaxis_title="Avg Temperature",

    font=dict(

        family="Courier New, monospace",

        size=9.5,

        color="#7f7f7f"

    )

)



fig.show()
import plotly.graph_objects as go

fig = go.Figure()



fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['AvgTemperature'],

    name="Actual AvgTemperature"

))





fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['mavg_12'],

    name="Moving Average AvgTemperature"

))



fig.add_trace(go.Scatter(

    x=emayd['Date'],

    y=emayd['ewm'],

    name="Exponential Smoothing AvgTemperature"

))



fig.update_layout(

    title="Avg Temp. - Actual vs vs Moving Average vs Exponential Smoothing of Delhi for the Month of April 2020",

    xaxis_title="Date",

    yaxis_title="Avg Temperature",

    font=dict(

        family="Courier New, monospace",

        size=9.5,

        color="#7f7f7f"

    )

)



fig.show()
df_months1 = df_months.reset_index()

from statsmodels.tsa.seasonal import seasonal_decompose



# Multiplicative Decomposition 

#result_mul = seasonal_decompose(df_months['AvgTemperature'], model='multiplicative', extrapolate_trend = 'freq', freq=12)



# Additive Decomposition

result_add = seasonal_decompose(df_months['AvgTemperature'], model='additive', extrapolate_trend = 'freq', freq=12)



# Plot

plt.rcParams.update({'figure.figsize': (10,10)})
#result_mul.plot().suptitle('Multiplicative Decompose', fontsize=15)
result_add.plot().suptitle('Additive Decompose', fontsize=15)
#vwap_itc_2020['mul_seasonal'] = result_mul.seasonal

#vwap_itc_2020['mul_trend'] = result_mul.trend

df_months['add_seasonal'] = result_add.seasonal

df_months['add_trend'] = result_add.trend
import plotly.graph_objects as go

fig = go.Figure()



df_monthsd = df_months.reset_index()



fig.add_trace(go.Scatter(

    x=df_monthsd['Date'],

    y=df_monthsd['add_seasonal'],

    name="Additive Seasonal Component"

))



fig.update_layout(

    title="Seasonal Component",

    xaxis_title="Date",

    yaxis_title="AvgTemperature",

    font=dict(

        family="Courier New, monospace",

        size=10.5,

        color="#7f7f7f"

    )

)



fig.show()
import plotly.graph_objects as go

fig = go.Figure()



df_monthsd = df_monthsd.reset_index()



fig.add_trace(go.Scatter(

    x=df_monthsd['Date'],

    y=df_monthsd['add_trend'],

    name="Additive Trend"

))



fig.update_layout(

    title="Trend Component",

    xaxis_title="Date",

    yaxis_title="AvgTemperature",

    font=dict(

        family="Courier New, monospace",

        size=10.5,

        color="#7f7f7f"

    )

)



fig.show()
#ACF plot to show auto-correlation upto lag of 20

    

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({'figure.figsize': (10,6)})

acf_plot = plot_acf(df_monthsd.AvgTemperature, lags=20)
plt.rcParams.update({'figure.figsize': (10,6)})



acf_plot = plot_pacf(df_monthsd.AvgTemperature, lags=20)
from statsmodels.tsa.arima_model import ARIMA

arima = ARIMA(df_monthsd.AvgTemperature[0:821].astype(np.float64), order = (1,0,0))

arima1 = ARIMA(df_monthsd.AvgTemperature[0:821].astype(np.float64), order = (0,0,0))

ar_model = arima.fit()

ar_model1 = arima1.fit()

ar_model.summary2()
forecast_arima = ar_model.predict(821, 849)

forecast_arima1 = ar_model1.predict(821, 849)



get_mape(df_monthsd.AvgTemperature[821:850].values, forecast_arima.values)
get_mape(df_monthsd.AvgTemperature[821:850].values, forecast_arima1.values)
arima = ARIMA(df_monthsd.AvgTemperature[0:821].astype(np.float64), order = (0,0,1))

ma_model = arima.fit()

ma_model.summary2()
forecast_arima = ma_model.predict(821, 849)



get_mape(df_monthsd.AvgTemperature[821:850].values, forecast_arima.values)
arima = ARIMA(df_monthsd.AvgTemperature[0:821].astype(np.float64), order = (1,0,1)) 

arma_model = arima.fit() 

arma_model.summary2()
forecast_arma = arma_model.predict(821, 849)

get_mape(df_monthsd.AvgTemperature[821:850].values, forecast_arma.values)
from statsmodels.tsa.stattools import adfuller



def adfuller_test(ts): 

    adfuller_result = adfuller(ts, autolag=None)

    adfuller_out = pd.Series(adfuller_result[0:4],

                             index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])

    print(adfuller_out)
adfuller_test(df_monthsd.AvgTemperature)
df_monthsd['vdiff'] = df_monthsd.AvgTemperature - df_monthsd.AvgTemperature.shift(1)

df_monthsd.head(5)

df_monthsd = df_monthsd.dropna()
import plotly.graph_objects as go

fig = go.Figure()



df_monthsd = df_monthsd.reset_index()



fig.add_trace(go.Scatter(

    x=df_monthsd['Date'],

    y=df_monthsd['vdiff'],

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
pacf_plot = plot_acf(df_monthsd.vdiff.dropna(), lags=10)
arima = ARIMA(df_monthsd.AvgTemperature[0:821].astype(np.float64), order = (1,1,1)) 

arima_model = arima.fit()

arima_model.summary2()
predict, stderr, ci = arima_model.forecast(steps=18)

get_mape(df_monthsd.AvgTemperature[821:850].values, predict)