import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn

import math
### load data



sales = pd.read_csv(

    '../input/transactions.csv', 

    parse_dates=['date'],

    dtype={'store_nbr':np.uint8, 'transactions':np.uint16}

)



items = pd.read_csv(

    '../input/items.csv',

    dtype={

        'item_nbr':np.uint32, 

        'class': np.uint16, 

        'perishable': np.bool

    }

)



stores = pd.read_csv(

    '../input/stores.csv',

)





oil = pd.read_csv(

    '../input/oil.csv',

    parse_dates=['date'],

    dtype={'dcoilwtico':np.float16}

)



holidays = pd.read_csv(

    '../input/holidays_events.csv',

    parse_dates=['date']

)



train = pd.read_csv(

    '../input/train.csv',

    nrows=6000000,

    parse_dates=['date'],

    dtype={

        'id':np.uint32,

        'store_nbr':np.uint8,

        'item_nbr': np.uint32,

        'onpromotion': np.bool,

        'unit_sales': np.float32

    }

)
train_items = train.merge(items, right_on='item_nbr', left_on='item_nbr', how='left')

train_items_stores = train_items.merge(stores, right_on='store_nbr', left_on='store_nbr', how='left')

train_items_stores_sales = train_items.merge(sales, right_on=['store_nbr', 'date'], left_on=['store_nbr', 'date'], how='left')
items.groupby(['family']).size().plot(kind='bar',stacked=True, figsize=(13,6),  grid=False)
def calc_percent(row):

    total = row.sum()

    percents = []

    for sales in row:

        if math.isnan(sales):

            percents.append(0.0)

        else:

            percents.append((sales/total) * 100)

    return percents
train_items_stores.groupby(

    ['type', 'family']

).size().unstack().apply(calc_percent, axis=1).plot(

    kind='bar', stacked=True, colormap= 'tab20c', figsize=(12,10),  grid=False)
train_items_stores.groupby(

    ['cluster', 'family']

).size().unstack().apply(calc_percent, axis=1).plot(

    kind='bar', stacked=True, colormap= 'tab20c', figsize=(12,10),  grid=False)
train_items_stores.groupby(

    ['state', 'family']

).size().unstack().apply(calc_percent, axis=1).plot(

    kind='bar', stacked=True, colormap= 'tab20c', figsize=(12,10),  grid=False)
train_items_stores.groupby(

    ['type', 'family']

).size().unstack().drop('GROCERY I', 1).apply(calc_percent, axis=1).plot(

    kind='bar', stacked=True, colormap= 'tab20c', figsize=(12,10),  grid=False)
train_items_stores.groupby(

    ['cluster', 'family']

).size().unstack().drop('GROCERY I', 1).apply(calc_percent, axis=1).plot(

    kind='bar', stacked=True, colormap= 'tab20c', figsize=(12,10),  grid=False)
train_items_stores.groupby(

    ['state', 'family']

).size().unstack().drop('GROCERY I', 1).apply(calc_percent, axis=1).plot(

    kind='bar', stacked=True, colormap= 'tab20c', figsize=(12,10),  grid=False)