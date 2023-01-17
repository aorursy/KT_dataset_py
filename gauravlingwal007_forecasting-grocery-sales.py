import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objs as go



import plotly.tools as tls

import plotly.offline as py

import plotly.graph_objs as go

import missingno as msno
holidays_events_df =pd.read_csv('../input/holidays_events.csv')

items_df =pd.read_csv('../input/items.csv')

oil_df =pd.read_csv('../input/oil.csv')

stores_df =pd.read_csv('../input/stores.csv')

transactions_df =pd.read_csv('../input/transactions.csv')

oil_df =pd.read_csv("../input/oil.csv",parse_dates=['date'],dtype={'dcoilwtico':np.float16})
oil_df['date'] = pd.to_datetime(oil_df['date'], format='%y-%m-%d')

oil_df['day_item_purchased'] = oil_df['date'].dt.day

oil_df['month_item_purchased'] =oil_df['date'].dt.month

oil_df['quarter_item_purchased'] = oil_df['date'].dt.quarter

oil_df['year_item_purchased'] = oil_df['date'].dt.year
plt.figure(figsize=(25,25))

plt.plot(oil_df['date'],oil_df['dcoilwtico'])

plt.show()
import calendar



transactions_df["year"] = transactions_df["date"].astype(str).str[:4].astype(np.int64)

transactions_df["month"] = transactions_df["date"].astype(str).str[5:7].astype(np.int64)

transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors ='coerce')

transactions_df['day_of_week'] = transactions_df['date'].dt.weekday_name





transactions_df["year"] = transactions_df["year"].astype(str)

transactions_df.head()
transactions1 = transactions_df.groupby('date')['transactions'].sum()

py.iplot([go.Scatter(

    x=transactions1.index,

    y=transactions1

)])
#month and year



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



x= transactions_df.groupby(['month','year'],as_index=False).agg({'transactions':'sum'})

y=x.pivot("month","year","transactions")

fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(y);
# analysis of what happening in day

x=transactions_df.groupby(['day_of_week', 'year'], as_index=False).agg({'transactions':'sum'})

y= x.pivot("day_of_week","year","transactions")

fig, ax = plt.subplots(figsize=(10,7))

sns.heatmap(y);

#people are buying more on saturday and sunday
set(stores_df.state)
regions_data = {    

             

    

        'state': ['Azuay',

 'Bolivar',

 'Chimborazo',

 'Cotopaxi',

 'El Oro',

 'Esmeraldas',

 'Guayas',

 'Imbabura',

 'Loja',

 'Los Rios',

 'Manabi',

 'Pastaza',

 'Pichincha',

 'Santa Elena',

 'Santo Domingo de los Tsachilas',

 'Tungurahua']}



df_regions = pd.DataFrame(regions_data, columns = ['state'])

df_regions_cities = pd.merge(df_regions, stores_df, on='state')



transactions_regions = pd.merge(transactions_df, df_regions_cities, on='store_nbr')

transactions_regions.head()
x= transactions_regions.groupby(['state','year'], as_index=False). agg({'transactions':'sum'})

y=x.pivot("state","year","transactions")

fig, ax = plt.subplots(figsize=(12,9))

sns.heatmap(y);

# Guayas and Pichincha, have more transactions in comparison to other states.
x= transactions_regions.groupby(['store_nbr','year'], as_index=False). agg({'transactions':'sum'})

y=x.pivot("store_nbr","year","transactions")

fig, ax = plt.subplots(figsize=(12,9))

sns.heatmap(y);

#store number 3,8,9,11,44,45,46,47,48,50 have more transactions.
items_df.head()
items_df.family.unique()
items_df_family =items_df.groupby(['family']).size().to_frame(name ='counts').reset_index()

items_df_family['percentage'] =items_df_family['counts']/items_df_family['counts'].sum() *100

items_df_family.head()
sns.set_style("white")

fig, ax =plt.subplots(figsize=(14,10))

ax = sns.barplot(x="percentage",y="family", data =items_df_family)

#Grocery has the maximum transactions.
types_dict = {'id':'int32',

             'item_nbr': 'int32',

             'store_nbr':'int8',

             'unit_sales':'float32',

             'onpromotion':'str',

             }
grocery_train =pd.read_csv('../input/train.csv', low_memory=True,dtype =types_dict,

                           converters={'unit_sales': lambda x:float(x) if float(x) > 0 else 0})

# log transform

#grocery_train["unit_sales"] = grocery_train["unit_sales"].apply(np.log1p)
# Calculate means

grocery_train = grocery_train.groupby(

    ['item_nbr', 'store_nbr', 'onpromotion']

)['unit_sales'].mean().to_frame('unit_sales')

# Inverse transform

#grocery_train["unit_sales"] = grocery_train["unit_sales"].apply(np.expm1)

#grocery_train
# Create submission

pd.read_csv(

    "../input/test.csv", usecols=[0, 2, 3, 4], dtype={'onpromotion': str}

).set_index(

    ['item_nbr', 'store_nbr', 'onpromotion']

).join(

    grocery_train, how='left'

).fillna(0).to_csv(

    'mean2.csv.gz', float_format='%.2f', index=None, compression="gzip"

)