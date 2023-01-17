# Importing the relevant libraries

import pandas as pd

import seaborn as sns

%matplotlib inline

import missingno as msno

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import numpy as np

from scipy.fftpack import fft

from matplotlib import pyplot as plt
items = pd.read_csv("../input/items.csv")

holiday_events = pd.read_csv("../input/holidays_events.csv")

stores = pd.read_csv("../input/stores.csv")

oil = pd.read_csv("../input/oil.csv")

transactions = pd.read_csv("../input/transactions.csv",parse_dates=['date'])

# I read in the full training data just to get prior information and here is the output:

# Output: "125,497,040 rows | 6 columns"

train = pd.read_csv("../input/train.csv", nrows=6000000)
train.head()
print("Nulls in Oil columns: {0} => {1}".format(oil.columns.values,oil.isnull().any().values))

print("="*70)

print("Nulls in holiday_events columns: {0} => {1}".format(holiday_events.columns.values,holiday_events.isnull().any().values))

print("="*70)

print("Nulls in stores columns: {0} => {1}".format(stores.columns.values,stores.isnull().any().values))

print("="*70)

print("Nulls in transactions columns: {0} => {1}".format(transactions.columns.values,transactions.isnull().any().values))
oil.isnull().any(axis=0).values
trace = go.Scatter(

    name='Oil prices',

    x=oil['date'],

    y=oil['dcoilwtico'].dropna(),

    mode='lines',

    line=dict(color='rgb(220, 150, 0, 0.8)'),

    #fillcolor='rgba(68, 68, 68, 0.3)',

    fillcolor='rgba(230, 200, 6, 0.3)',

    fill='tonexty' )



data = [trace]



layout = go.Layout(

    yaxis=dict(title='Daily Oil price'),

    title='Daily oil prices from Jan 2013 till July 2017',

    showlegend = False)

fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='pandas-time-series-error-bars')
stores.head(3)
# Unhide to see the sorted zip order

neworder = [23, 24, 26, 36, 41, 15, 29, 31, 32, 34, 39, 53, 4, 37, 40, 43, 8, 10, 19, 20, 33, 38, 13, 21, 2, 6, 7, 3, 22, 25, 27, 28, 30, 35, 42, 44, 48, 51, 16, 0, 1, 5, 52, 45, 46, 47, 49, 9, 11, 12, 14, 18, 17, 50]
nbr_cluster = stores.groupby(['store_nbr','cluster']).size()

nbr_cluster.unstack().iloc[neworder].plot(kind='bar',stacked=True, colormap= 'tab20', figsize=(13,11),  grid=False)

plt.title('Store numbers and the clusters they are assigned to')

plt.ylabel('')

plt.xlabel('Store number')

plt.show()
type_cluster = stores.groupby(['type','cluster']).size()

type_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'viridis_r', figsize=(13,11),  grid=False)

plt.title('Stacked Barplot of Store types and their cluster distribution')

plt.ylabel('Count of clusters in a particular store type')

plt.show()
city_cluster = stores.groupby(['city','type']).size()

city_cluster.unstack().plot(kind='bar',stacked=True, colormap= 'PuBu', figsize=(13,11),  grid=False)

plt.title('Stacked Barplot of Store types distributed across cities')

plt.ylabel('Count of stores in a particular city')

plt.show()
holiday_events.head(3)
holiday_local_type = holiday_events.groupby(['locale_name', 'type']).size()

holiday_local_type.unstack().plot(kind='bar',stacked=True, colormap= 'inferno', figsize=(12,10),  grid=False)

plt.title('Stacked Barplot of locale name against event type')

plt.ylabel('Count of entries')

plt.show()
x = holiday_events.groupby(['type', 'description']).size()

x.unstack().plot(kind='bar',stacked=True, colormap= 'inferno', figsize=(12,10),  grid=False)

plt.title('Stacked Barplot of locale name against event type')

plt.show()
print(transactions.head(3))

print("="*60)

print(transactions.shape)
transactions.iloc[33700]
plt.figure(figsize=(13,11))

plt.plot(transactions.date.values, transactions.transactions.values)

plt.axvline(x='2015-12-23',color='red',alpha=0.2)

plt.axvline(x='2016-12-23',color='red',alpha=0.2)

plt.axvline(x='2014-12-23',color='red',alpha=0.2)

plt.axvline(x='2013-12-23',color='red',alpha=0.2)

plt.ylim(-50, 10000)

plt.ylabel('transactions per day')

plt.xlabel('Date')

plt.show()
transactions.head()
items.head()
import sklearn

from sklearn import linear_model

from sklearn import model_selection

ridge = linear_model.Ridge()
data = train

(train, test) = model_selection.train_test_split(data, train_size=0.75)
ridge.fit(train[['store_nbr','item_nbr']], train['unit_sales'])
print(ridge.score(train[['store_nbr','item_nbr']], train['unit_sales']))

print(ridge.score(test[['store_nbr','item_nbr']], test['unit_sales']))
test_data = pd.read_csv('../input/test.csv')

test_data.head()
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission.head()
predictions = ridge.predict(test_data[['store_nbr','item_nbr']])

print(predictions)
sample_submission['unit_sales'] = predictions
sample_submission.to_csv('submission11.csv', index=False)
sub = pd.read_csv('../output/sumbmission1.csv')

sub.head()
%ls