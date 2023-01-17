import numpy as np

import pandas as pd

from fbprophet import Prophet
df = pd.read_excel('../input/included-names-of-columns/2020.xls')

print(df.head())

print(df.tail())

df[['Date', 'Values']] = df[['Date', 'Values']].values[::-1]

print(df)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import graph_objs as go



# инициализируем plotly

init_notebook_mode(connected = True)



# опишем функцию, которая будет визуализировать все колонки dataframe в виде line plot

def plotly_df(df, title = ''):

    data = []



    for column in df.columns:

        trace = go.Scatter(

            x = df['Date'],

            y = df['Values'],

            mode = 'lines',

            name = column

        )

        data.append(trace)



    layout = dict(title = title)

    fig = dict(data = data, layout = layout)

    iplot(fig, show_link=False)



plotly_df(df, title = 'Курс долару')
predictions = 31



# приводим dataframe к нужному формату

df.columns = ['ds', 'y']



# отрезаем из обучающей выборки последние 30 точек, чтобы измерить на них качество

train_df = df[:-predictions]



m = Prophet()

m.fit(train_df)



future = m.make_future_dataframe(periods=predictions)

forecast = m.predict(future)
m.plot(forecast)
m.plot_components(forecast)
cmp_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds'))

print(cmp_df)
cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']

cmp_df['p'] = 100*cmp_df['e']/cmp_df['y']

print ('MAPE', np.mean(abs(cmp_df[:-predictions]['p'])))

print ('MAE', np.mean(abs(cmp_df[:-predictions]['e'])))

print(cmp_df)