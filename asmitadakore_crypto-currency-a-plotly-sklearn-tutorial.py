# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import datetime as dt



import numpy as np

import pandas as pd



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import xgboost as xgb

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
df = pd.read_csv('../input/crypto-markets.csv')
df.info()
df = df.drop(['symbol', 'market'], axis=1)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['hlc_average'] = (df['high'] + df['low'] + df['close']) / 3

df['ohlc_average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df.head()
groupby = df.groupby('date', as_index=False).sum()

groupby.head()
trace0 = go.Scatter(

    x=groupby['date'], y=groupby['hlc_average'],

    name='HLC Average'

)



trace1 = go.Scatter(

    x=groupby['date'], y=groupby['volume'],

    name='Volume', yaxis='y2'

)



data = [trace0, trace1]

layout = go.Layout(

    title='General Overview',

    yaxis={

        'title': 'USD',

        'nticks': 10,

    },

    yaxis2={

        'title': 'Transactions',

        'nticks': 5,

        'showgrid': False,

        'overlaying': 'y',

        'side': 'right'

    }

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='time-series-overview')
df = df[df['date'] >= dt.date(2017, 1, 1)]
bitcoin = df[df['ranknow'] == 1]



others = df[(df['ranknow'] > 1) & (df['ranknow'] <= 10)]

others = others.groupby('date', as_index=False).mean()



minor = df[df['ranknow'] > 10]

minor = minor.groupby('date', as_index=False).mean()
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=(

    'Crypto Currency Price', 'Transaction Volume'

))



trace0 = go.Scatter(x=bitcoin['date'], y=bitcoin['hlc_average'], name='Bitcoin')

fig.append_trace(trace0, 1, 1)



trace1 = go.Scatter(x=bitcoin['date'], y=bitcoin['volume'], name='Bitcoin')

fig.append_trace(trace1, 1, 2)





trace2 = go.Scatter(x=others['date'], y=others['hlc_average'], name='Others')

fig.append_trace(trace2, 1, 1)



trace3 = go.Scatter(x=others['date'], y=others['volume'], name='Others')

fig.append_trace(trace3, 1, 2)



trace4 = go.Scatter(x=minor['date'], y=minor['hlc_average'], name='Minor ones')

fig.append_trace(trace4, 1, 1)



trace5 = go.Scatter(x=minor['date'], y=minor['volume'], name='Minor ones')

fig.append_trace(trace5, 1, 2)



fig['layout'].update(title='BitCoin vs others')

fig['layout'].update(showlegend=False)

fig['layout']['yaxis1'].update(title='USD')

fig['layout']['yaxis2'].update(title='Transactions')

fig['layout']['xaxis1'].update(nticks=6)

fig['layout']['xaxis2'].update(nticks=6)



py.iplot(fig, filename='bitcoin-vs-others')
top9 = df[(df['ranknow'] >= 2) & (df['ranknow'] <= 10)]

top9.name.unique()
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=(

    'Crypto Currency Price', 'Transaction Volume'

))



for name in top9.name.unique():

    crypto = top9[top9['name'] == name]

    trace0 = go.Scatter(x=crypto['date'], y=crypto['hlc_average'], name=name)

    fig.append_trace(trace0, 1, 1)

    

    trace1 = go.Scatter(x=crypto['date'], y=crypto['volume'], name=name)

    fig.append_trace(trace1, 1, 2)



fig['layout'].update(title='Other Crypto Currencies Comparison')

fig['layout'].update(showlegend=False)

fig['layout']['yaxis1'].update(title='USD')

fig['layout']['yaxis2'].update(title='Transactions')

fig['layout']['xaxis1'].update(nticks=6)

fig['layout']['xaxis2'].update(nticks=6)



py.iplot(fig, filename='other-crypto-currencies-comparison')
summary = top9.groupby('name', as_index=False).mean()

summary.sort_values('close', ascending=True)
low_price = top9[top9['ranknow'].isin([4, 6, 7, 9])]

low_price = low_price.groupby('date', as_index=False).mean()



high_price = top9[top9['ranknow'].isin([2, 3, 5, 8, 10])]

high_price = high_price.groupby('date', as_index=False).mean()
fig = tools.make_subplots(rows=1, cols=2, subplot_titles=(

    'Crypto Currency Price', 'Transaction Volume'

))



trace0 = go.Scatter(x=low_price['date'], y=low_price['hlc_average'], name='Low Price')

fig.append_trace(trace0, 1, 1)



trace1 = go.Scatter(x=low_price['date'], y=low_price['volume'], name='Low Price')

fig.append_trace(trace1, 1, 2)



trace2 = go.Scatter(x=high_price['date'], y=high_price['hlc_average'], name='High Price')

fig.append_trace(trace2, 1, 1)



trace3 = go.Scatter(x=high_price['date'], y=high_price['volume'], name='High Price')

fig.append_trace(trace3, 1, 2)



fig['data'][0].update(yaxis='y3')

fig['layout'].update(title='High vs Low Prices Comparison')

fig['layout'].update(showlegend=False)

fig['layout']['yaxis1'].update(title='USD')

fig['layout']['yaxis2'].update(title='Transactions')

fig['layout']['xaxis1'].update(nticks=6)

fig['layout']['xaxis2'].update(nticks=6)

fig['layout']['yaxis3'] = {

    'anchor': 'x1', 'domain': [0.0, 1.0], 'nticks': 6,

    'overlaying': 'y1', 'side': 'right', 'showgrid': False

}



py.iplot(fig, filename='high-vs-low-prices-comparison')
currency = df[df['name'] == 'Litecoin'].copy()

currency.head()
increasing_color = '#17BECF'

decreasing_color = '#7F7F7F'



data = []



layout = {

    'xaxis': {

        'rangeselector': {

            'visible': True

        }

    },

    # Adding a volume bar chart for candlesticks is a good practice usually

    'yaxis': {

        'domain': [0, 0.2],

        'showticklabels': False

    },

    'yaxis2': {

        'domain': [0.2, 0.8]

    },

    'legend': {

        'orientation': 'h',

        'y': 0.9,

        'yanchor': 'bottom'

    },

    'margin': {

        't': 40,

        'b': 40,

        'r': 40,

        'l': 40

    }

}



# Defining main chart

trace0 = go.Candlestick(

    x=currency['date'], open=currency['open'], high=currency['high'],

    low=currency['low'], close=currency['close'],

    yaxis='y2', name='Litecoin',

    increasing=dict(line=dict(color=increasing_color)),

    decreasing=dict(line=dict(color=decreasing_color)),

)



data.append(trace0)



# Adding some range buttons to interact

rangeselector = {

    'visible': True,

    'x': 0,

    'y': 0.8,

    'buttons': [

        {'count': 1, 'label': 'reset', 'step': 'all'},

        {'count': 6, 'label': '6 mo', 'step': 'month', 'stepmode': 'backward'},

        {'count': 3, 'label': '3 mo', 'step': 'month', 'stepmode': 'backward'},

        {'count': 1, 'label': '1 mo', 'step': 'month', 'stepmode': 'backward'},

    ]

}



layout['xaxis'].update(rangeselector=rangeselector)



# Setting volume bar chart colors

colors = []

for i, _ in enumerate(currency['date']):

    if i != 0:

        if currency['close'].iloc[i] > currency['close'].iloc[i-1]:

            colors.append(increasing_color)

        else:

            colors.append(decreasing_color)

    else:

        colors.append(decreasing_color)



trace1 = go.Bar(

    x=currency['date'], y=currency['volume'],

    marker=dict(color=colors),

    yaxis='y', name='Volume'

)



data.append(trace1)



# Adding Moving Average

def moving_average(interval, window_size=10):

    window = np.ones(int(window_size)) / float(window_size)

    return np.convolve(interval, window, 'same')



trace2 = go.Scatter(

    x=currency['date'][5:-5], y=moving_average(currency['close'])[5:-5],

    yaxis='y2', name='Moving Average',

    line=dict(width=1)

)



data.append(trace2)



# Adding boilinger bands

def bollinger_bands(price, window_size=10, num_of_std=5):

    rolling_mean = price.rolling(10).mean()

    rolling_std = price.rolling(10).std()

    upper_band = rolling_mean + (rolling_std * 5)

    lower_band = rolling_mean - (rolling_std * 5)

    return upper_band, lower_band



bb_upper, bb_lower = bollinger_bands(currency['close'])



trace3 = go.Scatter(

    x=currency['date'], y=bb_upper,

    yaxis='y2', line=dict(width=1),

    marker=dict(color='#ccc'), hoverinfo='none',

    name='Bollinger Bands',

    legendgroup='Bollinger Bands'

)

data.append(trace3)



trace4 = go.Scatter(

    x=currency['date'], y=bb_lower,

    yaxis='y2', line=dict(width=1),

    marker=dict(color='#ccc'), hoverinfo='none',

    name='Bollinger Bands', showlegend=False,

    legendgroup='Bollinger Bands'

)

data.append(trace4)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='litecoin-candlestick')
currency['target'] = currency['close'].shift(-30)
X = currency.dropna().copy()

X['year'] = X['date'].apply(lambda x: x.year)

X['month'] = X['date'].apply(lambda x: x.month)

X['day'] = X['date'].apply(lambda x: x.day)

X = X.drop(['date', 'slug', 'name', 'ranknow', 'target'], axis=1)



y = currency.dropna()['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



X_train.shape, X_test.shape
forecast = currency[currency['target'].isnull()]

forecast = forecast.drop('target', axis=1)



X_forecast = forecast.copy()

X_forecast['year'] = X_forecast['date'].apply(lambda x: x.year)

X_forecast['month'] = X_forecast['date'].apply(lambda x: x.month)

X_forecast['day'] = X_forecast['date'].apply(lambda x: x.day)

X_forecast = X_forecast.drop(['date', 'slug', 'name', 'ranknow'], axis=1)
currency = currency.drop('target', axis=1)
classifiers = {

    'LinearRegression': LinearRegression(),

    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=1),

    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=500)

}



summary = list()

for name, clf in classifiers.items():

    print(name)

    nada = clf.fit(X_train, y_train)

    

    print(f'R2: {r2_score(y_test, clf.predict(X_test)):.2f}')

    print(f'MAE: {mean_absolute_error(y_test, clf.predict(X_test)):.2f}')

    print(f'MSE: {mean_squared_error(y_test, clf.predict(X_test)):.2f}')

    print()

    

    summary.append({

        'MSE': mean_squared_error(y_test, clf.predict(X_test)),

        'MAE': mean_absolute_error(y_test, clf.predict(X_test)),

        'R2': r2_score(y_test, clf.predict(X_test)),

        'name': name,

    })
dtrain = xgb.DMatrix(X_train.values, y_train.values)

dtest = xgb.DMatrix(X_test.values)



param = {

    'max_depth': 10,

    'eta': 0.3

}

num_round = 20

bst = xgb.train(param, dtrain, num_round)

# make prediction

print('XGBoost')

print(f'R2: {r2_score(y_test, bst.predict(dtest)):.2f}')

print(f'MAE: {mean_absolute_error(y_test, bst.predict(dtest)):.2f}')

print(f'MSE: {mean_squared_error(y_test, bst.predict(dtest)):.2f}')



summary.append({

    'MSE': mean_squared_error(y_test, bst.predict(dtest)),

    'MAE': mean_absolute_error(y_test, bst.predict(dtest)),

    'R2': r2_score(y_test, bst.predict(dtest)),

    'name': 'XGBoost',

})
summary = pd.DataFrame(summary)



fig = tools.make_subplots(rows=1, cols=3, subplot_titles=(

    'R2', 'MAE', 'MSE'

))



trace0 = go.Bar(x=summary['name'], y=summary['R2'], name='R2')

fig.append_trace(trace0, 1, 1)



trace1 = go.Bar(x=summary['name'], y=summary['MAE'], name='MAE')

fig.append_trace(trace1, 1, 2)



trace2 = go.Bar(x=summary['name'], y=summary['MSE'], name='MSE')

fig.append_trace(trace2, 1, 3)



fig['layout'].update(title='Regression Metrics Comparison')

fig['layout'].update(showlegend=False)



py.iplot(fig, filename='regression-metrics-comparison')
clf = RandomForestRegressor(n_estimators=100, random_state=1)

clf.fit(X_train, y_train)

target = clf.predict(X_forecast)



final = pd.concat([currency, forecast])

final = final.groupby('date').sum()



day_one_forecast = currency.iloc[-1].date + dt.timedelta(days=1)

date = pd.date_range(day_one_forecast, periods=30, freq='D')

predictions = pd.DataFrame(target, columns=['target'], index=date)

final = final.append(predictions)

final.index.names = ['date']

final = final.reset_index()



trace0 = go.Scatter(

    x=final['date'], y=final['close'],

    name='Close'

)



trace1 = go.Scatter(

    x=final['date'], y=final['target'],

    name='Target'

)



data = [trace0, trace1]

layout = go.Layout(

    title='Prediction Visualization',

    yaxis={

        'title': 'USD',

        'nticks': 10,

    },

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='prediction-visualization')