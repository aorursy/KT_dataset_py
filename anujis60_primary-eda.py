# Importing the relevant libraries

import IPython.display

import pandas as pd

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from matplotlib import pyplot as plt
items = pd.read_csv("../input/items.csv")

holiday_events = pd.read_csv("../input/holidays_events.csv")

stores = pd.read_csv("../input/stores.csv")

oil = pd.read_csv("../input/oil.csv")

transactions = pd.read_csv("../input/transactions.csv",parse_dates=['date'])

train = pd.read_csv("../input/train.csv", parse_dates=['date'])
print(train.shape)

train.head()
# Check for NULL values in all files

print("Nulls in train: {0} => {1}".format(train.columns.values, train.isnull().any().values))

print('---')

print("Nulls in oil: {0} => {1}".format(oil.columns.values,oil.isnull().any().values))

print('---')

print("Nulls in holiday_events: {0} => {1}".format(holiday_events.columns.values,holiday_events.isnull().any().values))

print('---')

print("Nulls in stores: {0} => {1}".format(stores.columns.values,stores.isnull().any().values))

print('---')

print("Nulls in transactions: {0} => {1}".format(transactions.columns.values,transactions.isnull().any().values))
# EDA for oil.csv begins here #

oil.head(3)
# Plotting graph for oil price trends

trace = go.Scatter(

    name='Oil prices',

    x=oil['date'],

    y=oil['dcoilwtico'].dropna(),

    mode='lines',

   )



data = [trace]



layout = go.Layout(

    yaxis = dict(title = 'Daily Oil price'),

    showlegend = True)

fig = go.Figure(data = data, layout = layout)

py.iplot(fig, filename='pandas-time-series-error-bars')
# EDA for holiday_events.csv begins here #

holiday_events.head(3)
holiday_events.type.unique()
plt.style.use('seaborn-white')

holiday_local_type = holiday_events.groupby(['locale_name', 'type']).size()

holiday_local_type.unstack().plot(kind='bar',stacked=True, colormap= 'magma_r', figsize=(12,10),  grid=False)

plt.ylabel('Count of entries')

plt.show()
# EDA for items.csv begins here

items.head()
# BAR PLOT FOR ITEMS V/S FAMILY TYPE

x, y = (list(x) for x in zip(*sorted(zip(items.family.value_counts().index, 

                                         items.family.value_counts().values), 

                                        reverse = False)))

trace2 = go.Bar(

    y = items.family.value_counts().values,

    x = items.family.value_counts().index

)



layout = dict(

    title='Counts of items per family category',

     width = 900, height = 600,

    yaxis=dict(

        showgrid = True,

        showline = True,

        showticklabels = True

    ))



fig1 = go.Figure(data=[trace2])

fig1['layout'].update(layout)

py.iplot(fig1, filename='plots')
# Persihable or not 

plt.style.use('seaborn-white')

fam_perishable = items.groupby(['family', 'perishable']).size()

fam_perishable.unstack().plot(kind='bar',stacked=True, colormap = 'coolwarm', figsize=(12,10),  grid = True)

plt.ylabel('count')

plt.show()
# EDA for stores.csv begins here #

stores.head(3)
# store distribution across states

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

ax = sns.countplot(y = stores['state'], data = stores) 
# store distribution across cities

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

ax = sns.countplot(y = stores['city'], data = stores) 
# Unique state names

stores.state.unique()
# Unique state names

stores.city.unique()
# Various types of stores and their count

fig, ax = plt.subplots()

fig.set_size_inches(8, 5)

ax = sns.countplot(x = "type", data = stores, palette="Paired")
ct = pd.crosstab(stores.state, stores.type)

ct.plot.bar(figsize = (12, 6), stacked=True)

plt.legend(title='type')

plt.show()
ct = pd.crosstab(stores.city, stores.type)



ct.plot.bar(figsize = (12, 6), stacked=True)

plt.legend(title='type')



plt.show()
# total no. of unique stores 

stores.store_nbr.nunique()
# total no. of stores (including non-unique)

stores.cluster.sum()
fig, ax = plt.subplots()

fig.set_size_inches(12, 7)

ax = sns.countplot(x = "cluster", data = stores)
# EDA for transactions.csv begins here #

transactions.head(3)
# Finding out the no.of transactions (rows)

print("There are {0} transactions".

      format(transactions.shape[0], transactions.shape[1]))
# Time series plot for transaction #

plt.style.use('seaborn-white')

plt.figure(figsize=(13,11))

plt.plot(transactions.date.values, transactions.transactions.values, color='grey')

plt.axvline(x='2015-12-23',color='red',alpha=0.3)

plt.axvline(x='2016-12-23',color='red',alpha=0.3)

plt.axvline(x='2014-12-23',color='red',alpha=0.3)

plt.axvline(x='2013-12-23',color='red',alpha=0.3)

plt.axvline(x='2013-05-12',color='green',alpha=0.2, linestyle= '--')

plt.axvline(x='2015-05-10',color='green',alpha=0.2, linestyle= '--')

plt.axvline(x='2016-05-08',color='green',alpha=0.2, linestyle= '--')

plt.axvline(x='2014-05-11',color='green',alpha=0.2, linestyle= '--')

plt.axvline(x='2017-05-14',color='green',alpha=0.2, linestyle= '--')

plt.ylabel('Transactions per day', fontsize= 16)

plt.xlabel('Date', fontsize= 16)

plt.show()