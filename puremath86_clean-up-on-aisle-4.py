import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

train.head()
print(test.shape)

test.head()
test.tail()
test.date.unique()
sub = pd.read_csv('../input/sample_submission.csv')

print(sub.shape)

sub.head()
stores = pd.read_csv('./../input/stores.csv')

print(stores.shape)

stores.head()
items = pd.read_csv('./../input/items.csv')

print(items.shape)

items.head()
oil = pd.read_csv('./../input/oil.csv')

print(oil.shape)

oil.head()
transactions = pd.read_csv('./../input/transactions.csv')

print(transactions.shape)

transactions.head()
store = train[train.store_nbr == 1]
store.head()
store.item_nbr.unique()
example = store[store.item_nbr == 103665][['date', 'unit_sales']]



example.plot( figsize=(13, 8))
example['cum_sales'] = example.unit_sales.cumsum()



example.plot( figsize=(13, 8))
def get_store(store):

    df_store = train[train.store_nbr == store][['date', 'item_nbr', 'unit_sales']]

    return df_store





def store_itemize(df_store, item):

    df_item = df_store[df_store.item_nbr == item][['date', 'unit_sales']]

    df_item.index = pd.DatetimeIndex(df_item.date)

    idx = pd.date_range('2013-01-09', '2017-08-31')

    df_item = df_item.reindex(idx, fill_value=df_item.unit_sales.mean())

    del df_item['date']

    df_item['cum_sum'] = df_item.unit_sales.cumsum()

    return df_item





df_store = get_store(1)

df_item = store_itemize(df_store, 103665)

df_item.head()



# for store in train.store_nbr.unique():

#     df_store = get_store(store)

#     for item in df_store.item_nbr.unique():

#         df_item = store_itemize(df_store, item)
df_item.tail()