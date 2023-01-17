import pandas as pd

import numpy as np
holidays_events_df = pd.read_csv('../input/holidays_events.csv', low_memory=False)

items_df = pd.read_csv('../input/items.csv', low_memory=False)

oil_df = pd.read_csv('../input/oil.csv', low_memory=False)

stores_df = pd.read_csv('../input/stores.csv', low_memory=False)

transactions_df = pd.read_csv('../input/transactions.csv', low_memory=False)
import calendar



transactions_df["year"] = transactions_df["date"].astype(str).str[:4].astype(np.int64)

transactions_df["month"] = transactions_df["date"].astype(str).str[5:7].astype(np.int64)

transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors ='coerce')

transactions_df['day_of_week'] = transactions_df['date'].dt.weekday_name





transactions_df["year"] = transactions_df["year"].astype(str)

transactions_df.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



x = transactions_df.groupby(['month', 'year'], as_index=False).agg({'transactions':'sum'})

y = x.pivot("month", "year", "transactions")

fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(y);
x = transactions_df.groupby(['day_of_week', 'year'], as_index=False).agg({'transactions':'sum'})

y = x.pivot("day_of_week", "year", "transactions")

fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(y);
regions_data = {

        

        'region': ['Sierra','Sierra','Sierra','Sierra', 'Sierra', 'Sierra', 'Sierra', 'Sierra', 

                   'Costa', 'Costa', 'Costa', 'Costa', 'Costa', 'Costa' , 'Costa' , 'Oriente'],

        

    

        'state': ['Imbabura','Tungurahua', 'Pichincha', 'Azuay', 'Bolivar', 'Chimborazo', 

                 'Loja', 'Cotopaxi', 'Esmeraldas', 'Manabi', 'Santo Domingo de los Tsachilas', 

                 'Santa Elena', 'Guayas', 'El Oro', 'Los Rios', 'Pastaza']}



df_regions = pd.DataFrame(regions_data, columns = ['region', 'state'])



df_regions_cities = pd.merge(df_regions, stores_df, on='state')





transactions_regions = pd.merge(transactions_df, df_regions_cities, on='store_nbr')

transactions_regions.head()
x = transactions_regions.groupby(['state', 'year'], as_index=False).agg({'transactions':'sum'})

y = x.pivot("state", "year", "transactions")

fig, ax = plt.subplots(figsize=(12,9))

sns.heatmap(y);
x = transactions_regions.groupby(['store_nbr', 'year'], as_index=False).agg({'transactions':'sum'})

y = x.pivot("store_nbr", "year", "transactions")

fig, ax = plt.subplots(figsize=(12,9))

sns.heatmap(y);
items_df.head()
items_df.family.unique()
items_df_family = items_df.groupby(['family']).size().to_frame(name = 'count').reset_index()

items_df_family['percentage']= items_df_family['count']/items_df_family['count'].sum() * 100

items_df_family.head()
sns.set_style("white")

fig, ax =plt.subplots(figsize=(14,10))

ax = sns.barplot(x="percentage", y="family", data=items_df_family, palette="BuGn_r")
dtypes = {'store_nbr': np.dtype('int64'),

          'item_nbr': np.dtype('int64'),

          'unit_sales': np.dtype('float64'),

          'onpromotion': np.dtype('O')}





train = pd.read_csv('../input/train.csv', index_col='id', parse_dates=['date'], dtype=dtypes)

date_mask = (train['date'] >= '2016-01-01') & (train['date'] <= '2016-12-31') & (train['store_nbr'] == 25)

train = train[date_mask]

train.head()
df_train_item = pd.merge(train, items_df, on='item_nbr').sort_values(by='date')

df_train_item["year"] = df_train_item["date"].astype(str).str[:4].astype(np.int64)

df_train_item["month"] = df_train_item["date"].astype(str).str[5:7].astype(np.int64)

df_train_item.head()
sns.set_style("white")

ax = plt.subplots(figsize=(13, 9))

sns.countplot(x="family", hue="month", data=df_train_item, palette="Greens_d",

              order=df_train_item.family.value_counts().iloc[:7].index);
df_train_item['item_nbr'].value_counts().nlargest(30)