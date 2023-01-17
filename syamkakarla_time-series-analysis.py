!pip install pmdarima
import pandas as pd
import numpy as np
import plotly as py
import datetime
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from plotly.offline import iplot, init_notebook_mode

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
# from statsmodels.tsa.statespace import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima.utils import ndiffs
from numpy import log

# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
path = '/kaggle/input/corona-virus-report/'

data = pd.read_csv(path+'covid_19_clean_complete.csv', index_col=False)

data.drop('Province/State', axis=1, inplace=True)
data['dt'] = data.Date
data.Date = pd.to_datetime(data.Date)
data.set_index('Date', inplace=True)

data.head()
x = data.groupby('Country/Region')['Confirmed'].max().sort_values(ascending=[False])[:15]
x = pd.DataFrame({'country': x.index, 'values': x.values})
px.bar(x, x = 'country', y = 'values', color = 'country', title='Top 15 Countries Affected by COVID-19')
def plot_data(data, *args):
    for country in args:
        df = data[data['Country/Region'] == country]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df.Confirmed, mode='lines', name='Confirmed'))
        fig.add_trace(go.Scatter(x=df.index, y= df.Deaths, mode='lines', name='Deaths'))
        fig.add_trace(go.Scatter(x=df.index, y=df.Recovered, mode='lines', name='Recovered'))

        fig.update_layout(title_text= country+" Details")
        fig.show()
plot_data(data, 'India', 'US', 'Russia')
df = data[data['Country/Region'] == 'India']
df.head()
df.info()
result = adfuller(df.Confirmed.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
fig, axes = plt.subplots(3, 2, sharex=True, figsize=(9, 7))
axes[0,0].plot(df.Confirmed.dropna().values)
axes[1,0].plot(df.Confirmed.diff().dropna().values)
axes[2,0].plot(df.Confirmed.diff().diff().dropna().values)

# axes[1, 1].set(ylim=(0,20))
plot_acf(df.Confirmed, lags = df.Confirmed.shape[0]-2, ax = axes[0,1])
plot_acf(df.Confirmed.diff().dropna(), ax = axes[1,1], lags = df.Confirmed.shape[0]-3,)
plot_acf(df.Confirmed.diff().diff().dropna(), ax = axes[2,1], lags = df.Confirmed.shape[0]-4,)
plt.show()
## Adf Test
y = df.Confirmed.values
print(ndiffs(y, test='adf')  # 2

# KPSS test
,ndiffs(y, test='kpss')  # 2

# PP test:
,ndiffs(y, test='pp'))  # 2
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Confirmed.diff().dropna().values); 
axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.5))
# axes[1].set(xlim=(0,20))
plot_pacf(df.Confirmed.diff().dropna(), ax=axes[1])

plt.show()
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Confirmed.diff().dropna().values); 
axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.5))
# axes[1].set(xlim=(0,20))
plot_acf(df.Confirmed.diff().dropna(), ax=axes[1])

plt.show()
# 1,1,2 ARIMA Model
model = ARIMA(df.Confirmed.values, order=(1,2,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2, figsize=(12,2))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()
train = df.Confirmed[ :100]
test = df.Confirmed[100: ]
n_periods = test.shape[0]

a_model = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(a_model.summary())

fc, confint = a_model.predict(n_periods=n_periods, return_conf_int=True)
# ind = test.index + datetime.timedelta(days= abs(test.shape[0] - fc_series.shape[0]))
plt.figure(figsize=(10, 3))
plt.plot(train.index, train.values, label='Trian')
plt.plot(test.index, test.values,label='Test')
plt.plot( test.index,fc, color='green', label = 'predicted')
plt.legend()
plt.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='train'))
fig.add_trace(go.Scatter(x=test.index, y= test.values, mode='lines', name='test'))
fig.add_trace(go.Scatter(x=test.index, y=fc, mode='lines', name='predicted'))

fig.update_layout(title_text= "Prediction of the model")
fig.show()
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)
model = pm.auto_arima(df.Confirmed, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())
model.plot_diagnostics(figsize=(10,6))
plt.show()
# Forecast
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(df.Confirmed.dropna().shape[0], df.Confirmed.dropna().shape[0]+n_periods)
days = df.index + datetime.timedelta(days= n_periods)
days = days.strftime('%d-%m-%Y').values

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df.Confirmed.values, label='Data')
# plt.xticks(range(days.shape[0]), new_xticks, rotation=90, horizontalalignment='right')
plt.plot(fc_series, color='orange', label = 'Forecast')
plt.fill_between(lower_series.index,
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of Confirmed Cases in India ")
plt.legend()
plt.show()
model = pm.auto_arima(df.Deaths, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=1,              # frequency of series
                      d=True,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(10,6))
plt.show()

# Forecast
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(df.Deaths.dropna().shape[0], df.Deaths.dropna().shape[0]+n_periods)
days = df.index + datetime.timedelta(days= n_periods)
days = days.strftime('%d-%m-%Y').values

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df.Deaths.values, label='Data')
# plt.xticks(range(days.shape[0]), new_xticks, rotation=90, horizontalalignment='right')
plt.plot(fc_series, color='orange', label = 'Forecast')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of Deaths in India ")
plt.legend()
plt.show()
model = pm.auto_arima(df.Recovered, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=1,              # frequency of series
                      d=True,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(10,6))
plt.show()

# Forecast
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(df.Recovered.dropna().shape[0], df.Recovered.dropna().shape[0]+n_periods)
days = df.index + datetime.timedelta(days= n_periods)
days = days.strftime('%d-%m-%Y').values

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df.Recovered.values, label='Data')
# plt.xticks(range(days.shape[0]), new_xticks, rotation=90, horizontalalignment='right')
plt.plot(fc_series, color='orange', label = 'Forecast')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of Recovered in India ")
plt.legend()
plt.show()
