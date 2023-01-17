import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns # advanced plotting

from scipy import stats

from datetime import datetime

from sklearn import preprocessing

from sklearn.model_selection import KFold

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
Q = pd.read_csv('../input/stock-market-wide-datasets/000000000000Q')

AM = pd.read_csv('../input/stock-market-wide-datasets/AM')

EVENT = pd.read_csv('../input/stock-market-wide-datasets/event')

NEWS = pd.read_csv('../input/stock-market-wide-datasets/news')
print(Q.head())

print(AM.head())

print(EVENT.head())

print(NEWS.head())
print(Q.info())

print(AM.info())

print(EVENT.info())

print(NEWS.info())
Q['time'] = pd.to_datetime(Q['time'])

AM['time'] = pd.to_datetime(AM['time'])

EVENT['system_time'] = pd.to_datetime(EVENT['system_time'])

NEWS['datetime'] = pd.to_datetime(NEWS['datetime'])
print(Q['ticker'].unique())

print(AM['symbol'].unique())

print(EVENT['symbol'].unique())

print(NEWS['stock'].unique())
Qticker = Q['ticker'].unique().tolist()

AMsymbol = AM['symbol'].unique().tolist()
Q_ticker_set = {}

for t in Qticker:

    Q_ticker_set[t] = Q[Q['ticker']==t].sort_values(by='time')

AM_symbol_set = {}

for t in AMsymbol:

    AM_symbol_set[t] = AM[AM['symbol']==t].sort_values(by='time')
Q_AAPL = Q_ticker_set['AAPL']

Q_AAPL.pop('ticker')

Q_AAPL.head()
AM_AAPL = AM_symbol_set['AAPL']

AM_AAPL.pop('symbol')

AM_AAPL.head()
print(Q.describe())

print(AM.describe())

print(EVENT.describe())

print(NEWS.describe())
plt.figure(figsize=(10,10))

plt.bar(Q_AAPL.bid_price,AM_AAPL.average_price)

plt.xticks(rotation=90)

plt.show()