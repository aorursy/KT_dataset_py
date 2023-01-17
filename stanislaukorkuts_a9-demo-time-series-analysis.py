import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import os



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import graph_objs as go

import requests

import pandas as pd



print(__version__) # need 1.9.0 or greater

init_notebook_mode(connected = True)





def plotly_df(df, title = ''):

    data = []

    

    for column in df.columns:

        trace = go.Scatter(

            x = df.index,

            y = df[column],

            mode = 'lines',

            name = column

        )

        data.append(trace)

    

    layout = dict(title = title)

    fig = dict(data = data, layout = layout)

    iplot(fig, show_link=False)
df = pd.read_csv('../input/wiki_machine_learning.csv', sep = ' ')

df = df[df['count'] != 0]

df.head()
df.shape
df.date = pd.to_datetime(df.date)
plotly_df(df.set_index('date')[['count']])
from fbprophet import Prophet
predictions = 30



df = df[['date', 'count']]

df.columns = ['ds', 'y']

df.tail()
train_df = df[:-predictions]

train_df.tail()
proph = Prophet()

proph.fit(train_df)
future = proph.make_future_dataframe(periods=predictions)

future.tail()
forecast = proph.predict(future)

print(f"Prediction of the number of views of the wiki page on January 20 is {round(float(forecast.tail(1)['yhat']))}")
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(df[-predictions:]['y'], forecast[-predictions:]['yhat'])

print(f"Mean absolute error is {round(mae)}")
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(df[-predictions:]['y'], forecast[-predictions:]['yhat'])

print(f"Mean absolute percentage error is {round(mape, 1)}")
%matplotlib inline

import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (15, 10)
p_value = sm.tsa.stattools.adfuller(train_df['y'].values)[1]

print(f'p_value = {p_value}')

if p_value > 0.05:

    print(f'series is not stationary')

else:

    print(f'series is stationary')
plotly_df(df.set_index('ds'))
timeseries = df.set_index('ds')

timeseries.head()
newseries = timeseries.copy()

newseries['y'], lmbda = stats.boxcox(timeseries['y'])

plotly_df(newseries)

sm.tsa.stattools.adfuller(newseries['y'].values)[1]
#Построение ACF для определения параметра q

sm.graphics.tsa.plot_acf(newseries['y'].values,lags=100);
#Построение PACF для определения параметра p

sm.graphics.tsa.plot_pacf(newseries['y'].values,lags=100);
#Нужно избавиться от сезонности

diff_season = newseries.diff(7).dropna()

diff_season.plot()
#Проверка на стационарность

sm.tsa.stattools.adfuller(diff_season['y'].values)[1]
#Построение ACF для определения параметра Q

sm.graphics.tsa.plot_acf(diff_season['y'].values,lags=100);
#Построение PACF для определения параметра P

sm.graphics.tsa.plot_pacf(diff_season['y'].values,lags=100);
ps = range(0, 4)

d=0

qs = range(0, 2)

Ps = range(0, 3)

D=0

Qs = range(0, 1)



from itertools import product



parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

len(parameters_list)
%%time

results = []

best_aic = float("inf")

warnings.filterwarnings('ignore')



for param in parameters_list:

    #try except нужен, потому что на некоторых наборах параметров модель не обучается

    try:

        model=sm.tsa.statespace.SARIMAX(timeseries, order=(param[0], d, param[1]), 

                                        seasonal_order=(param[2], D, param[3], 7)).fit()

    #выводим параметры, на которых модель не обучается и переходим к следующему набору

    except:

        #print('wrong parameters:', param)

        continue

    aic = model.aic

    #сохраняем лучшую модель, aic, параметры

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results.append([param, model.aic])

    

warnings.filterwarnings('default')



result_table = pd.DataFrame(results)

result_table.columns = ['parameters', 'aic']
result_table[result_table['aic'] == result_table['aic'].min()]