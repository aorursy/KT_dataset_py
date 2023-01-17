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
m = Prophet()

m.fit(train_df)
future = m.make_future_dataframe(periods=predictions)
forecast = m.predict(future)
m.plot_components(forecast)
m.plot(forecast)
cmp_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds'))
cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']

cmp_df['p'] = 100*cmp_df['e']/cmp_df['y']

print ('MAPE', np.mean(abs(cmp_df[-predictions:]['p'])))

print ('MAE', np.mean(abs(cmp_df[-predictions:]['e'])))
def show_forecast(cmp_df, num_predictions, num_values):

    # верхняя граница доверительного интервала прогноза

    upper_bound = go.Scatter(

        name='Upper Bound',

        x=cmp_df.tail(num_predictions).index,

        y=cmp_df.tail(num_predictions).yhat_upper,

        mode='lines',

        marker=dict(color="blue"),

        line=dict(width=0),

        fillcolor='rgba(68, 68, 68, 0.3)',

        fill='tonexty')



    # прогноз

    forecast = go.Scatter(

        name='Prediction',

        x=cmp_df.tail(predictions).index,

        y=cmp_df.tail(predictions).yhat,

        mode='lines',

        line=dict(color='rgb(31, 119, 180)'),

    )



    # нижняя граница доверительного интервала

    lower_bound = go.Scatter(

        name='Lower Bound',

        x=cmp_df.tail(num_predictions).index,

        y=cmp_df.tail(num_predictions).yhat_lower,

        marker=dict(color="blue"),

        line=dict(width=0),

        mode='lines')



    # фактические значения

    fact = go.Scatter(

        name='Fact',

        x=cmp_df.tail(num_values).index,

        y=cmp_df.tail(num_values).y,

        marker=dict(color="red"),

        mode='lines',

    )



    # последовательность рядов в данном случае важна из-за применения заливки

    data = [lower_bound, upper_bound, forecast, fact]



    layout = go.Layout(

        yaxis=dict(title='Посты'),

        title='Опубликованные посты на Хабрахабре',

        showlegend = False)



    fig = go.Figure(data=data, layout=layout)

    iplot(fig, show_link=False)



show_forecast(cmp_df, predictions, 200)
%matplotlib inline

import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (15, 10)
sm.tsa.seasonal_decompose(train_df['y'].values, freq=7).plot();

print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(train_df['y'])[1])
train_df.set_index('ds', inplace=True)
ps = range(0, 2)

ds = range(0, 2)

qs = range(0, 4)

Ps = range(0, 4)

Ds = range(0, 3)

Qs = range(0, 2)
from itertools import product



parameters = product(ps, ds, qs, Ps, Ds, Qs)

parameters_list = list(parameters)

len(parameters_list)
import warnings

from tqdm import tqdm

results1 = []

best_aic = float("inf")

warnings.filterwarnings('ignore')



for param in tqdm(parameters_list):

    try:

        model=sm.tsa.statespace.SARIMAX(train_df['y'], order=(param[0], param[1], param[2]), 

                                        seasonal_order=(param[3], param[4], param[5], 7)).fit(disp=-1)

    except (ValueError, np.linalg.LinAlgError):

        continue

    aic = model.aic

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results1.append([param, model.aic])
result_table1 = pd.DataFrame(results1)

result_table1.columns = ['parameters', 'aic']

print(result_table1.sort_values(by = 'aic', ascending=True).head())
result_table1[result_table1['parameters'].isin([(1, 0, 2, 3, 1, 0),

                                                (1, 1, 2, 3, 2, 1),

                                                (1, 1, 2, 3, 1, 1),

                                                (1, 0, 2, 3, 0, 0)])]
import scipy.stats

train_df['y_box'], lmbda = scipy.stats.boxcox(train_df['y']) 

print("The optimal Box-Cox transformation parameter: %f" % lmbda)
results2 = []

best_aic = float("inf")



for param in tqdm(parameters_list):

    #try except is necessary, because on some sets of parameters the model can not be trained

    try:

        model=sm.tsa.statespace.SARIMAX(train_df['y_box'], order=(param[0], param[1], param[2]), 

                                        seasonal_order=(param[3], param[4], param[5], 7)).fit(disp=-1)

    #print parameters on which the model is not trained and proceed to the next set

    except (ValueError, np.linalg.LinAlgError):

        continue

    aic = model.aic

    #save the best model, aic, parameters

    if aic < best_aic:

        best_model = model

        best_aic = aic

        best_param = param

    results2.append([param, model.aic])

    

warnings.filterwarnings('default')
result_table2 = pd.DataFrame(results2)

result_table2.columns = ['parameters', 'aic']

print(result_table2.sort_values(by = 'aic', ascending=True).head())
result_table2[result_table2['parameters'].isin([(1, 0, 2, 3, 1, 0),

                                                (1, 1, 2, 3, 2, 1),

                                                (1, 1, 2, 3, 1, 1),

                                                (1, 0, 2, 3, 0, 0)])].sort_values(by='aic')
print(best_model.summary())
plt.subplot(211)

best_model.resid[13:].plot()

plt.ylabel(u'Residuals')



ax = plt.subplot(212)

sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)



print("Student's test: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])

print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])
def invboxcox(y,lmbda):

    # reverse Box Cox transformation

    if lmbda == 0:

        return(np.exp(y))

    else:

        return(np.exp(np.log(lmbda * y + 1) / lmbda))
train_df['arima_model'] = invboxcox(best_model.fittedvalues, lmbda)



train_df.y.tail(200).plot()

train_df.arima_model[13:].tail(200).plot(color='r')

plt.ylabel('wiki pageviews');