import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go
root = '../input/m5-forecasting-accuracy'



calendar = pd.read_csv(root + '/calendar.csv')

sales_train_valid = pd.read_csv(root + '/sales_train_validation.csv')

sell_prices = pd.read_csv(root + '/sell_prices.csv')



submission = pd.read_csv(root + '/sample_submission.csv')



print('Size of calendar', calendar.shape)

print('Size of sales_train_valid', sales_train_valid.shape)

print('Size of sell_prices', sell_prices.shape)
calendar.head()
sales_train_valid.head()
sell_prices.head()
submission.head()
# Missing data

def missing_data(data):

    total = data.isnull().sum()

    percent = (total/data.isnull().count()*100)

    tp = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'], sort=True)

    

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tp['Types'] = types

    

    return tp
missing_data(calendar).head(8)
sales_train_valid['state_id'].unique()
sales_train_valid['cat_id'].unique()
sales_train_valid['item_id'].unique()
def most_frequent_values(col):

    total = col.count()

    itm = col.value_counts().index[0]

    val = col.value_counts().values[0]

    percent = np.round(val / total * 100, 3)

    dic = {'Total': total, 'Most Frequent Item': itm, 'Value': val, 'Percent': percent}

    return dic
col = sales_train_valid['cat_id']

most_frequent_values(col)
col = sales_train_valid['state_id']

most_frequent_values(col)
sell_prices['store_id'].unique()
sell_prices['item_id'].unique()
sell_prices.loc[sell_prices['item_id'] == 'FOODS_3_827']
# view a single item states

sales_region = sales_train_valid.loc[sales_train_valid['item_id'] == 'FOODS_3_827']
pd.crosstab(sales_region['state_id'], sales_region['store_id'])
sales_region.head()
fig = go.Figure()

for i in range(10):

    fig.add_trace(go.Scatter(x=None, y=sales_region.iloc[i, 6:].values,

                        mode='lines',

                        name=sales_region.iloc[i, 5]))

fig.update_layout(title="FOODS_3_827 sales")

fig.show()
fig = go.Figure()

for i in range(10):

    fig.add_trace(go.Scatter(x=None, y=sales_region.iloc[i, 6:].rolling(30).mean().values,

                        mode='lines',

                        name=sales_region.iloc[i, 5]))

fig.update_layout(title="FOODS_3_827 sales, rolling mean 30 days")

fig.show()
fig = go.Figure()

for i in range(10):

    fig.add_trace(go.Scatter(x=None, y=sales_region.iloc[i, 6:].rolling(100).mean().values,

                        mode='lines',

                        name=sales_region.iloc[i, 5]))

fig.update_layout(title="FOODS_3_827 sales, rolling mean 100 days")

fig.show()
sell_prices.loc[sell_prices['store_id'] == 'CA_1']
sales_train_valid.loc[sales_train_valid['store_id'] == 'CA_1']
ca_1_sales = sales_train_valid.loc[sales_train_valid['store_id'] == 'CA_1']

pd.crosstab(ca_1_sales['cat_id'], ca_1_sales['dept_id'])
ca_1_sales['dept_id'].unique()
fig = go.Figure()

for dep in ca_1_sales['dept_id'].unique():

    fig.add_trace(go.Scatter(x=None, y=ca_1_sales.loc[ca_1_sales['dept_id'] == dep].rolling(30).mean().values,

                        mode='lines',

                        name=dep))

fig.update_layout(title="CA_1 sales of dep, rolling mean 30 days")

fig.show()
ca_1_sales['cat_id'].unique()
fig = go.Figure()

for cat in ca_1_sales['cat_id'].unique():

    fig.add_trace(go.Scatter(x=None, y=ca_1_sales.loc[ca_1_sales['cat_id'] == cat].rolling(30).mean().values,

                        mode='lines',

                        name=cat))

fig.update_layout(title="CA_1 sales of cat, rolling mean 30 days")

fig.show()