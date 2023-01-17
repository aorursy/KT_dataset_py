# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_covid_basic = pd.read_csv('/kaggle/input/covid19dataset/COVID_Data_Basic.csv')
df_covid = pd.read_csv('/kaggle/input/covid19dataset/COVID_Data.csv')
df_ibov = pd.read_csv('/kaggle/input/ibovespa-points/datasets_Ibovespa_points_1994_2020.csv')
df_brazil_covid = df_covid_basic[df_covid_basic['Country']=='Brazil']
def convert_date(date):
    splitted_date = date.split(".")
    return splitted_date[2] + "-"+ splitted_date[1] + "-" + splitted_date[0]

def remove_m_volume(str):
    if 'M' in str:
        str = str.replace("M",'')
        str = str.replace(',','')
        return float(str) * 1000000
    elif 'K' in str:
        str = str.replace("K",'')
        str = str.replace(',','')
        return float(str) * 1000
    else:
        return 0
    
df_ibov['Date'] = df_ibov['Date'].apply(lambda x : convert_date(x))
df_ibov['Volume'] = df_ibov['Volume'].apply(lambda x : remove_m_volume(x))
df_ibov['Volume'] = df_ibov['Volume'].astype(float)

df_covid_period_ibov = df_ibov[df_ibov['Date']>='2020-01-01']
df_brazil_covid = df_brazil_covid[df_brazil_covid['Date'].isin(list(df_covid_period_ibov['Date']))]
df_covid_period_ibov = df_covid_period_ibov.drop(df_covid_period_ibov.index[[0,1]])
df_brazil_covid = df_brazil_covid.sort_values(by='Date', ascending=True)
df_covid_period_ibov = df_covid_period_ibov.sort_values(by='Date', ascending=True)
df_covid_period_ibov['Close points'] = df_covid_period_ibov['Close points'].apply(lambda x : x.replace(',','.'))
df_covid_period_ibov['Close points'] = df_covid_period_ibov['Close points'].astype(float)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
from plotly.graph_objs import *
import plotly.graph_objects as go

def plot_line_chart(df, date_column1, column2, title):
    fig = px.line(x=pd.to_datetime(df[date_column1].values), y=df[column2])
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.show()

def covid_charts(df, column1, column2, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'].values, y=df[column1],
        name=column1,
        mode='lines',
        marker_color='rgba(152, 0, 0, .8)'
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'].values, y=df[column2],
        name=column2,
        mode='lines',
        marker_color='rgba(255, 182, 193, .9)'
    ))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.show()

plot_line_chart(df_covid_period_ibov, 'Date', 'Close points', 'Ibovespa index in covid period')
covid_charts(df_brazil_covid,'Confirmed', 'Death', 'Confirmed cases vs Deaths')
covid_charts(df_brazil_covid,'newConfirmed', 'newDeath', 'Daily confirmed cases vs Daily new deaths')
def plot_line_chart_rolling_mensure(df, column2, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'].values, y=df[column2].rolling(window=7, center=False).mean(),
        name='rolling mean '+ column2,
        mode='lines',
        marker_color='rgba(255, 0, 0, .8)'
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'].values, y=df[column2].rolling(window=7, center=False).std(),
        name='rolling std ' + column2,
        mode='lines',
        marker_color='rgba(0, 255, 0, .9)'
    ))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.show()
    
plot_line_chart_rolling_mensure(df_covid_period_ibov, 'Close points', 'Ibovespa index in covid period')
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(df_covid_period_ibov['Close points'].values)
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

detrend = difference(df_covid_period_ibov['Close points'].values)
frame = pd.DataFrame([])
frame['detrend'] = detrend

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=frame['detrend'].values,
    name='rolling mean ',
    mode='lines',
    marker_color='rgba(0, 0, 255, .8)'
))
fig.add_trace(go.Scatter(
    y=frame['detrend'].rolling(window=7, center=False).mean(),
    name='rolling mean ',
    mode='lines',
    marker_color='rgba(255, 0, 0, .8)'
))
fig.add_trace(go.Scatter(
    y=frame['detrend'].rolling(window=7, center=False).std(),
    name='rolling std ' ,
    mode='lines',
    marker_color='rgba(0, 255, 0, .9)'
))
fig.update_layout({
    'title': 'detrend series',
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()

adf_test(detrend)
deseasonality = difference(df_covid_period_ibov['Close points'].values, 7)
frame = pd.DataFrame([])
frame['detrend'] = deseasonality

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=frame['detrend'].values,
    name='rolling mean ',
    mode='lines',
    marker_color='rgba(0, 0, 255, .8)'
))
fig.add_trace(go.Scatter(
    y=frame['detrend'].rolling(window=7, center=False).mean(),
    name='rolling mean ',
    mode='lines',
    marker_color='rgba(255, 0, 0, .8)'
))
fig.add_trace(go.Scatter(
    y=frame['detrend'].rolling(window=7, center=False).std(),
    name='rolling std ' ,
    mode='lines',
    marker_color='rgba(0, 255, 0, .9)'
))
fig.update_layout({
    'title': 'deseasonality series',
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()

adf_test(deseasonality)
from statsmodels.tsa.stattools import acf, pacf
acf_values = acf(detrend)
pacf_values = pacf(detrend, method='ols')
def compute_limit(serie, upper=True):
    limit = []
    y=1.96/np.sqrt(len(serie))
    for i in range(0, len(serie)):
        if upper == True:
            limit.append(y)
        else:
            limit.append(-y)
    return limit

def plot_acf(serie, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=serie,
        name='serie',
        mode='lines',
        marker_color='rgba(0, 0, 255, .8)'
    ))
    fig.add_trace(go.Scatter(
        y=compute_limit(serie),
        name='upper limit',
        mode='lines',
        marker_color='rgba(0, 0, 0, .8)'
    ))
    fig.add_trace(go.Scatter(
        y=compute_limit(serie, False),
        name='down limit' ,
        mode='lines',
        marker_color='rgba(0, 0, 0, .9)'
    ))
    fig.update_layout({
        'title': title,
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.show()  
plot_acf(acf_values, 'Autocorrelation function')
plot_acf(pacf_values, 'Partial Autocorrelation function')
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_covid_period_ibov['Close points'].values, order=(1,0,1)) 
model_fit = model.fit(disp=0)
from pandas import datetime
forecast = model_fit.predict()
fig = go.Figure()
fig.add_trace(go.Scatter(
        y=forecast,
        name='predict',
        mode='lines',
        marker_color='rgba(0, 0, 255, .8)'
))
fig.add_trace(go.Scatter(
        y=df_covid_period_ibov['Close points'].values,
        name='Original',
        mode='lines',
        marker_color='rgba(0, 0, 0, .8)'
))
fig.update_layout({
        'title': 'Predict vs Real',
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show() 
import seaborn as sns

def heatmap(df, df2,column1, column2, title): 
    matrix = np.corrcoef(df[column1].values, df2[column2].values)
    ax = sns.heatmap(matrix, annot=True, fmt="f", 
                     yticklabels=[column1, column2], 
                     xticklabels=[column1, column2])
    ax.set_title(title)

heatmap(df_covid_period_ibov, df_brazil_covid, 'Close points', 'Death', 'correlation between Ibovespa points and Death by corona')
heatmap(df_covid_period_ibov, df_brazil_covid, 'Close points', 'Confirmed', 
        'correlation between Ibovespa points and total confirmed cases')
heatmap(df_covid_period_ibov, df_brazil_covid, 'Close points', 'newDeath', 'correlation between Ibovespa points and daily deaths')
heatmap(df_covid_period_ibov, df_brazil_covid, 'Close points', 'newConfirmed', 'correlation between Ibovespa points and daily new cases')
heatmap(df_covid_period_ibov, df_brazil_covid, 'Close points', 'Recovered', 'correlation between Ibovespa points and recover')
heatmap(df_covid_period_ibov, df_brazil_covid, 'Close points', 'newRecovered', 'correlation between Ibovespa points and daily recovered')
