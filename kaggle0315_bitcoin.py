# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import datetime

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from sklearn.metrics import confusion_matrix

from fbprophet import Prophet

from fbprophet.plot import plot_plotly

from plotly.offline import init_notebook_mode, iplot

from plotly import figure_factory as FF

from plotly.subplots import make_subplots

sns.set()

%matplotlib inline



init_notebook_mode(connected=True)
def dateparse (time_in_secs):    

    return datetime.datetime.fromtimestamp(float(time_in_secs))





# bitstampとcoinbaseは取引所の名前

bitstamp = pd.read_csv("../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv", parse_dates=True, date_parser=dateparse, index_col=[0])

coinbase = pd.read_csv("../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv", parse_dates=True, date_parser=dateparse, index_col=[0])
bitstamp['2016-6-23'][datetime.time(12, 30):datetime.time(12, 40)]
bitstamp.loc[datetime.datetime(2016, 6, 23, 12, 36, 0), 'Low'] = bitstamp.loc[datetime.datetime(2016, 6, 23, 12, 37, 0), 'Open']

bitstamp.loc[datetime.datetime(2016, 6, 23, 12, 36, 0), 'Close'] = bitstamp.loc[datetime.datetime(2016, 6, 23, 12, 37, 0), 'Open']
display(bitstamp.head())

display(bitstamp.tail())
display(coinbase.head())

display(coinbase.tail())
bitstamp.describe()
bitstamp.info()
bitstamp.isnull().sum()
fill_OHLC = np.delete(np.insert(bitstamp['Close'].fillna(method='ffill').values, 0, 0), -1)

fill_WP = bitstamp['Weighted_Price'].fillna(method='ffill').values

fill_df = pd.DataFrame({'Open': fill_OHLC,

                        'High': fill_OHLC,

                        'Low': fill_OHLC,

                        'Close': fill_OHLC,

                        'Volume_(BTC)': np.zeros(len(bitstamp)),

                        'Volume_(Currency)': np.zeros(len(bitstamp)),

                        'Weighted_Price': fill_WP},

                       index = bitstamp.index)

bitstamp.fillna(fill_df, inplace=True)

bitstamp.isnull().sum()
resample_d = bitstamp.resample('D').agg({'Open': 'first',

                                       'High': 'max',

                                       'Low': 'min',

                                       'Close': 'last',

                                       'Volume_(BTC)': 'sum',

                                       'Volume_(Currency)': 'sum',

                                       'Weighted_Price': 'mean'})



ma_period = [5,25,75]  # 1週間、1ヶ月、3ヶ月

for ma in ma_period:

    column_name = 'MA{}'.format(ma)

    resample_d[column_name] = pd.Series.rolling(resample_d['Close'],ma).mean()
resample_w = bitstamp.resample('W').agg({'Open': 'first',

                                       'High': 'max',

                                       'Low': 'min',

                                       'Close': 'last',

                                       'Volume_(BTC)': 'sum',

                                       'Volume_(Currency)': 'sum',

                                       'Weighted_Price': 'mean'})



ma_period = [13,26,52]  # 3ヶ月、6ヶ月、1年

for ma in ma_period:

    column_name = 'MA{}'.format(ma)

    resample_w[column_name] = pd.Series.rolling(resample_w['Close'],ma).mean()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.7, 0.3])

fig.add_trace(go.Candlestick(x=resample_d.index, open=resample_d.Open, high=resample_d.High, low=resample_d.Low, close=resample_d.Close, showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=resample_d.index, y=resample_d.MA5.values, mode='lines', name='MA5'), row=1, col=1)

fig.add_trace(go.Scatter(x=resample_d.index, y=resample_d.MA25.values, mode='lines', name='MA25'), row=1, col=1)

fig.add_trace(go.Scatter(x=resample_d.index, y=resample_d.MA75.values, mode='lines', name='MA75'), row=1, col=1)

fig.add_trace(go.Bar(x=resample_d.index, y=resample_d['Volume_(BTC)'], marker_color='rgb(26, 118, 255)', name='Volume'), row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False)

iplot(fig)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.7, 0.3])

fig.add_trace(go.Candlestick(x=resample_w.index, open=resample_w.Open, high=resample_w.High, low=resample_w.Low, close=resample_w.Close, showlegend=False), row=1, col=1)

fig.add_trace(go.Scatter(x=resample_w.index, y=resample_w.MA13.values, mode='lines', name='MA13'), row=1, col=1)

fig.add_trace(go.Scatter(x=resample_w.index, y=resample_w.MA26.values, mode='lines', name='MA26'), row=1, col=1)

fig.add_trace(go.Scatter(x=resample_w.index, y=resample_w.MA52.values, mode='lines', name='MA52'), row=1, col=1)

fig.add_trace(go.Bar(x=resample_w.index, y=resample_w['Volume_(BTC)'], marker_color='rgb(26, 118, 255)', name='Volume'), row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False)

iplot(fig)
resample_d['Daily Return'] = resample_d['Close'].pct_change()

print(f'変化率の平均:{round(resample_d["Daily Return"].mean(), 4)}')

print(f'変化率の標準偏差:{round(resample_d["Daily Return"].std(), 4)}')
fig = go.Figure(data=go.Scatter(x=resample_d.index, y=resample_d['Daily Return']))

iplot(fig)
plt.figure(figsize=(20, 8))

sns.distplot(resample_d['Daily Return'].dropna(), bins=100, kde=False)



plt.text(np.percentile(resample_d['Daily Return'].dropna(), 2.5), 300, "{:0.5f}".format(np.percentile(resample_d['Daily Return'].dropna(), 2.5)), ha="center", va="center")  # 下側2.5%点

plt.axvline(x=np.percentile(resample_d['Daily Return'].dropna(), 2.5), linewidth=2, color='r')

plt.text(np.percentile(resample_d['Daily Return'].dropna(), 97.5), 300, "{:0.5f}".format(np.percentile(resample_d['Daily Return'].dropna(), 97.5)), ha="center", va="center",)  # 上側2.5%点

plt.axvline(x=np.percentile(resample_d['Daily Return'].dropna(), 97.5), linewidth=2, color='r')
by_dayofweek = pd.DataFrame(resample_d['Daily Return'].dropna())

by_dayofweek['day_of_week'] = by_dayofweek.index.dayofweek

day_of_week = {0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

by_dayofweek['day_of_week'].replace(day_of_week, inplace=True)



sns.catplot(x='day_of_week', y='Daily Return', data=by_dayofweek, kind='point', order=['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'], height=6, aspect=2)
by_month = pd.DataFrame(resample_d['Daily Return'].dropna())

by_month['month'] = by_month.index.month



sns.catplot(x='month', y='Daily Return', data=by_month, kind='point', height=6, aspect=2)
def ret_shift(shift_period):

    ret = pd.DataFrame(resample_d['Daily Return'])

    for i in range(1, shift_period+1):

        ret[f'DR_shift_{i}'] = ret.shift(i)['Daily Return']

    return ret
ret_7 = ret_shift(7)





def Threshold(x):

    if x > 0.01:

        return 1

    elif -0.01 < x < 0.01:

        return 0

    else:

        return -1

    



ret_7['Daily Return'] = ret_7['Daily Return'].map(Threshold)

ret_7.dropna(inplace=True)
X = ret_7.drop('Daily Return', axis=1)

y = ret_7['Daily Return']

labels = y.unique()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



parameters = {

    'n_estimators': [10, 25, 50, 75, 100],

    'random_state': [0],

    'min_samples_split': [2, 5, 10, 15, 20, 25, 30],

    'max_depth': [5, 10, 15, 20, 25, 30]

}



gs = GridSearchCV(estimator=RandomForestClassifier(),

                 param_grid=parameters,

                 cv=3,

                 n_jobs=-1)



gs.fit(X_train, y_train)



model = gs.best_estimator_



print(gs.best_score_)

print(gs.score(X_test, y_test))
y_pred = gs.best_estimator_.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels), columns=labels, index=labels)
pd.DataFrame({'特徴': X_train.columns, 'importance':model.feature_importances_}).sort_values('importance',ascending=False)