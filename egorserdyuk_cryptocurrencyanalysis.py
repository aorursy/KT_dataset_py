# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import datetime as dt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.express as px

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import RFE, SelectKBest, chi2, VarianceThreshold



import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/all-crypto-currencies/crypto-markets.csv")
df.info()
df.describe()
dfnum = df.drop(['symbol', 'slug', 'name', 'date'], axis=1)
dfnum.mean()  # Mean value
def dataframe_range(dataframe):  # Data range

    df_range = pd.DataFrame(dataframe.max() - dataframe.min())

    return df_range
dataframe_range(dfnum)
dfnum.std()  # Standard deviation
dfnum.std() ** 2  # Dispersion is squared degree of standard deviation
df.isnull().sum()  # Checking NULLs, we're lucky
df = df.drop(['slug'], axis=1)  # Drop useless columns



df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  # Transform date to date object
df.head(10)
df['ohlc_average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
top10 = df[(df['ranknow'] >= 1) & (df['ranknow'] <= 10)]

top10.name.unique()
fig = px.pie(top10, values='volume', names='name', title='Cryptocurrencies Top-10 by Transaction Volume')

fig.show()
fig = px.pie(top10, values='market', names='name', title='Cryptocurrencies Top-10 by Market capitalization')

fig.show()
fig = tools.make_subplots(subplot_titles=('Time'))

for name in top10.name.unique():

    currency = top10[top10['name'] == name]

    trace = go.Scatter(x=currency['date'], y=currency['ohlc_average'], name=name)

    fig.append_trace(trace, 1, 1)

    

fig['layout'].update(title='Top-10 Cryptocurrencies exchange rates comparison')

fig['layout']['yaxis1'].update(title='USD')

fig.show()
top10minorCurrencies = df[(df['ranknow'] >= 11) & (df['ranknow'] <= 21)]



top10minorCurrencies.name.unique()
fig = px.pie(top10minorCurrencies, values='volume', names='name', title='Minor Cryptocurrencies by Transaction Volume')

fig.show()
fig = px.pie(top10minorCurrencies, values='market', names='name', title='Minor Cryptocurrencies by Market capitalization')

fig.show()
fig = tools.make_subplots(subplot_titles=('Time'))

for name in top10minorCurrencies.name.unique():

    currency = top10minorCurrencies[top10minorCurrencies['name'] == name]

    trace = go.Scatter(x=currency['date'], y=currency['ohlc_average'], name=name)

    fig.append_trace(trace, 1, 1)

    

fig['layout'].update(title='Top-10 Cryptocurrencies exchange rates comparison')

fig['layout']['yaxis1'].update(title='USD')

fig.show()
currency = df[df['name']=='Bitcoin']

currency.head()
currency['target'] = currency['close'].shift(-30)
X = currency.dropna().copy()

X['year'] = X['date'].apply(lambda x: x.year)

X['month'] = X['date'].apply(lambda x: x.month)

X['day'] = X['date'].apply(lambda x: x.day)

X = X.drop(['date', 'symbol', 'name', 'ranknow', 'target'], axis=1)



y = currency.dropna()['target']



X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=1)



X_train.shape, X_test.shape
forecast = currency[currency['target'].isnull()]

forecast = forecast.drop('target', axis=1)



X_forecast = forecast.copy()

X_forecast['year'] = X_forecast['date'].apply(lambda x: x.year)

X_forecast['month'] = X_forecast['date'].apply(lambda x: x.month)

X_forecast['day'] = X_forecast['date'].apply(lambda x: x.day)

X_forecast = X_forecast.drop(['date', 'symbol', 'name', 'ranknow'], axis=1)
currency = currency.drop('target', axis=1)
classifiers = {

    'LinearRegression': LinearRegression(),

    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=1)

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
import xgboost as xgb



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

    'R-квадратичная ошибка', 'Средняя абсолютная ошибка', 'Среднеквадратичная ошибка'

))



trace0 = go.Bar(x=summary['name'], y=summary['R2'], name='R2')

fig.append_trace(trace0, 1, 1)



trace1 = go.Bar(x=summary['name'], y=summary['MAE'], name='MAE')

fig.append_trace(trace1, 1, 2)



trace2 = go.Bar(x=summary['name'], y=summary['MSE'], name='MSE')

fig.append_trace(trace2, 1, 3)



fig['layout'].update(title='Сравнение метрик')

fig['layout'].update(showlegend=False)



py.iplot(fig)
clf = LinearRegression()

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

    title='Визуализация результатов предсказания курса BTC по линейной регрессии',

    yaxis={

        'title': 'USD',

        'nticks': 10,

    },

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
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

    title='Визуализация результатов предсказания курса BTC по случайным деревьям',

    yaxis={

        'title': 'USD',

        'nticks': 10,

    },

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
dtrain = xgb.DMatrix(X_train.values, y_train.values)

dtest = xgb.DMatrix(X_test.values)



param = {

    'max_depth': 10,

    'eta': 0.3

}

num_round = 20

bst = xgb.train(param, dtrain, num_round)



X_forecast = xgb.DMatrix(X_forecast.values)

target = bst.predict(X_forecast)



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

    title='Визуализация результатов предсказания курса BTC по xgboost',

    yaxis={

        'title': 'USD',

        'nticks': 10,

    },

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)