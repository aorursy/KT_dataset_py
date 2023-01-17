# Time series analysis of monthly UFO reports between 1996 and 2013

# Decomposition into seasonal, trend and residual components

# Autocorrelation, ARIMA and SARIMAX model for forecasting





import numpy as np

import pandas as pd

from scipy import signal

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import acf, pacf

import statsmodels.tsa as tsa 

import statsmodels.api as sm

import plotly as py

import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()
# load UFO data

ufo_data = pd.read_csv('../input/scrubbed.csv', usecols=[0, 4, 5, 7], low_memory=False)

ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'], errors='coerce')
df1 = ufo_data
# copy dataframe

df1.insert(1, 'year', df1['datetime'].dt.year)

df1['year'] = df1['year'].fillna(0).astype(int)

df1.insert(1, 'month', df1['datetime'].dt.month)

df1['month'] = df1['month'].fillna(0).astype(int)

df1.insert(1, 'day', df1['datetime'].dt.month)

df1['day'] = df1['day'].fillna(0).astype(int)
3# Format and group UFO rows by month, cut missing data from end

df1t = df1[(df1['datetime'] >= '1996-01-01') & (df1['datetime'] <= '2013-01-01')]

daily = df1t.set_index('datetime').groupby(pd.TimeGrouper(freq='D'))['comments'].count()

monthly = df1t.set_index('datetime').groupby(pd.TimeGrouper(freq='M'))['comments'].count()

yearly = df1t.set_index('datetime').groupby(pd.TimeGrouper(freq='A'))['comments'].count()

daily.sort_index(ascending=True)

monthly.sort_index(ascending=True)

yearly.sort_index(ascending=True)
# reset index

mm = monthly.reset_index()

yy = yearly.reset_index()

dd = daily.reset_index()

Ym = mm.comments

Yy = yy.comments

Yd = dd.comments
Y = monthly
# Plot data to have a look

# Create traces

trace01 = go.Scatter(

    x = daily.index,

    y = Yd,

    mode = 'line',

    name = 'daily',

    marker = dict(

        size = 7,

        color = '#50C5B7',

        line = dict(

            width = 2,

            color = '#50C5B7'

        ) 

        )

)  

trace02 = go.Scatter(

    x = monthly.index,

    y = Ym,

    mode = 'lines',

    name = 'monthly',

    marker = dict(

        size = 7,

        color = '#6184D8',

        line = dict(

            width = 2,

            color = '#6184D8',

        ) 

        )

)

layout1 = go.Layout(

         title = 'UFO Reports 1964 - 2013',

         xaxis = dict(

             rangeslider = dict(thickness = 0.05),

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             #range = [0, 10000],

             showline = True,

             showgrid = False)

         )



data = [trace01,trace02]

fig = dict(data=data, layout=layout1)

iplot(fig)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=12,center=False).mean()

    rolstd = timeseries.rolling(window=12,center=False).std()



    #Plot rolling statistics:

    trace01 = {'x': timeseries.index, 'y': timeseries, 'type': 'scatter', 'name': 'Original'}

    trace02 = {'x': rolmean.index, 'y': rolmean, 'type': 'scatter', 'name': 'Rolling mean'}

    trace03 = {'x': rolstd.index, 'y': rolstd, 'type': 'scatter', 'name': 'Rolling Std' }



    

    layout1 = go.Layout(

         title = 'Rolling Mean & Standard Deviation',

         #marker=dict(colorscale="Viridis"),

         xaxis = dict(

            rangeslider = dict(thickness = 0.05),

            showline = True,

         ),

         yaxis = dict(

             #range = [0, 10000],

             showline = True,

             showgrid = False,

        )

        )



    data = [trace01,trace02,trace03]

    fig = dict(data=data, layout=layout1)

    iplot(fig)



    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
test_stationarity(Y)
Y_log = np.log(Y)

Y_log

Y_log = Y_log.replace([np.inf, -np.inf], 0)

Y_log.dropna(inplace=True)

moving_avg = Y_log.rolling(window=12,center=False).mean()

test_stationarity(Y_log)
Y_log_moving_avg_diff = Y_log - moving_avg

Y_log_moving_avg_diff.dropna(inplace=True)

test_stationarity(Y_log_moving_avg_diff)
expweighted_avg = Y_log.ewm(halflife=12,ignore_na=False,min_periods=0,adjust=True).mean()

Y_log_ewma_diff = Y_log - expweighted_avg

Y_log_diff = Y_log - Y_log.shift()

Y_log_diff.dropna(inplace=True)

test_stationarity(Y_log_ewma_diff)
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(Y_log)

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid
# Plot decomposed timeseries

fig = tls.make_subplots(rows=4, cols=1, shared_xaxes=True)

fig.append_trace({'x': Y_log.index, 'y': Y_log, 'type': 'scatter', 'name': 'Raw'}, 1, 1)

fig.append_trace({'x': Y_log.index, 'y': seasonal, 'type': 'scatter', 'name': 'Seasonal'}, 2, 1)

fig.append_trace({'x': Y_log.index, 'y': residual, 'type': 'scatter', 'name': 'Residual'}, 3, 1)

fig.append_trace({'x': Y_log.index, 'y': trend, 'type': 'scatter', 'name': 'Trend'}, 4, 1)



layout1 = go.Layout(

         title = 'UFO Reports - timeseries decomposition',

         xaxis = dict(

             rangeslider = dict(thickness = 0.05),

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             #range = [0, 10000],

             showline = True,

             showgrid = False)

         )



fig['layout'].update(title='UFO Reports - timeseries decomposition')

iplot(fig)
Y_log_decompose = residual

Y_log_decompose.dropna(inplace=True)

test_stationarity(Y_log_decompose)
# # Autocorrelation 

lag_acf = acf(Y_log_diff, nlags=11)

lag_pacf = pacf(Y_log_diff, nlags=11, method='ols')

## axis lines

# plt.axhline(y=-1.96/np.sqrt(len(Y_log_diff)),linestyle='--',color='gray')

# plt.axhline(y=1.96/np.sqrt(len(Y_log_diff)),linestyle='--',color='gray')

trace1 = go.Scatter(

    x = range(0,len(lag_acf)),

    y = lag_acf,

    mode = 'lines',

    name = 'Autocorrelation',

    marker = dict(

        size = 4,

        color = '#6184D8',

        line = dict(

            width = 2,

            color = '#6184D8',

        ) 

        )

)

trace2 = go.Scatter(

    x = range(0,len(lag_acf)),

    y = lag_pacf,

    mode = 'lines',

    name = 'Partial autocorrelation',

    marker = dict(

        size = 4,

        color = '#50C5B7',

        line = dict(

            width = 2,

            color = '#50C5B7',

        ) 

        )

)



layout = go.Layout(

    title='Autocorrelation function',

    xaxis=dict(

        title='Lag',

        titlefont=dict(

            family='Arial',

            size=16,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='Corr',

        titlefont=dict(

            family='Arial',

            size=16,

            color='#7f7f7f'

        )

    )

) 

data = [trace1,trace2]

fig = go.Figure(data=data, layout=layout)

#iplot(fig) works in python 2.7
# Fit the AR model

model = ARIMA(Y_log, order=(2, 1, 0))  

results_AR = model.fit(disp=-1)  



trace1 = go.Scatter(

    x = Y_log_diff.index,

    y = Y_log_diff,

    mode = 'lines',

    name = 'Y_log',

    marker = dict(

        size = 4,

        color = '#6184D8',

        line = dict(

            width = 2,

            color = '#6184D8',

        ) 

        )

)

trace2 = go.Scatter(

    x = Y_log_diff.index,

    y = results_AR.fittedvalues,

    mode = 'lines',

    name = 'AR fitted values',

    marker = dict(

        size = 4,

        color = '#50C5B7',

        line = dict(

            width = 2,

            color = '#50C5B7',

        ) 

        )

)

layout = go.Layout(

    title='AR Model',

    xaxis=dict(

        title='Date',

        titlefont=dict(

        )

    ),

    yaxis=dict(

        title='',

        titlefont=dict(

        )

    )

) 



data = [trace1,trace2]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
model = ARIMA(Y_log, order=(0, 1, 2))  

results_MA = model.fit(disp=-1)  



trace1 = go.Scatter(

    x = Y_log_diff.index,

    y = Y_log_diff,

    mode = 'lines',

    name = 'Y_log',

    marker = dict(

        size = 4,

        color = '#6184D8',

        line = dict(

            width = 2,

            color = '#6184D8',

        ) 

        )

)

trace2 = go.Scatter(

    x = Y_log_diff.index,

    y = results_MA.fittedvalues,

    mode = 'lines',

    name = 'MA fitted values',

    marker = dict(

        size = 4,

        color = '#50C5B7',

        line = dict(

            width = 2,

            color = '#50C5B7',

        ) 

        )

)

layout = go.Layout(

    title='MA Model',

    xaxis=dict(

        title='Date',

        titlefont=dict(

        )

    ),

    yaxis=dict(

        title='',

        titlefont=dict(

        )

    )

) 



data = [trace1,trace2]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#ARIMA

model = ARIMA(Y_log, order=(2, 1, 2))  

results_ARIMA = model.fit(disp=-1)  



trace1 = go.Scatter(

    x = Y_log_diff.index,

    y = Y_log_diff,

    mode = 'lines',

    name = 'Y_log',

    marker = dict(

        size = 4,

        color = '#6184D8',

        line = dict(

            width = 2,

            color = '#6184D8',

        ) 

        )

)

trace2 = go.Scatter(

    x = Y_log_diff.index,

    y = results_ARIMA.fittedvalues,

    mode = 'lines',

    name = 'ARIMA fitted values',

    marker = dict(

        size = 4,

        color = '#50C5B7',

        line = dict(

            width = 2,

            color = '#50C5B7',

        ) 

        )

)

layout = go.Layout(

    title='ARIMA Model',

    xaxis=dict(

        title='Date',

        titlefont=dict(

        )

    ),

    yaxis=dict(

        title='',

        titlefont=dict(

        )

    )

) 



data = [trace1,trace2]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(Y_log.ix[0], index=monthly.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)*trend

pa = predictions_ARIMA

pa = (pa-min(pa))/(max(pa)-min(pa))

Yn = (Y-min(Y))/(max(Y)-min(Y))



trace1 = go.Scatter(

    x = Y_log_diff.index,

    y = Y,

    mode = 'lines',

    name = 'Observed',

    marker = dict(

        size = 4,

        color = '#6184D8',

        line = dict(

            width = 2,

            color = '#6184D8',

        ) 

        )

)

trace2 = go.Scatter(

    x = Y_log_diff.index,

    y = predictions_ARIMA,

    mode = 'lines',

    name = 'Predicted values',

    marker = dict(

        size = 4,

        color = '#50C5B7',

        line = dict(

            width = 2,

            color = '#50C5B7',

        ) 

        )

)

layout = go.Layout(

    title='ARIMA Model forecast',

    xaxis=dict(

        title='Date',

        titlefont=dict(

        )

    ),

    yaxis=dict(

        title='',

        titlefont=dict(

        )

    )

) 



data = [trace1,trace2]

fig = go.Figure(data=data, layout=layout)

iplot(fig)