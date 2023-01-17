import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import squarify #TreeMap

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
etrade_df = pd.read_csv('../input/india-trade-data/2018-2010_export.csv')

itrade_df = pd.read_csv('../input/india-trade-data/2018-2010_import.csv')
etrade_df.head()
etrade_df.tail()
itrade_df.head()
itrade_df.tail()
etrade_df.info()
itrade_df.info()
etrade_df.describe()
itrade_df.describe()
etrade_df.isnull().sum()
itrade_df.isnull().sum()
etrade_df[etrade_df.value==0].head(5)
itrade_df[itrade_df.value==0].head()
etrade_df[etrade_df.country == "UNSPECIFIED"].head(5)
itrade_df[itrade_df.country == "UNSPECIFIED"].head(5)
# Replace the missing datas of the value column with their means grouped by the commodity



# export data

etrade_df["value"].fillna(etrade_df.groupby('Commodity')['value'].transform('mean'),inplace = True)

# import data

itrade_df["value"].fillna(itrade_df.groupby('Commodity')['value'].transform('mean'),inplace = True)
etrade_df['value'].isnull().sum()
itrade_df['value'].isnull().sum()
yearly_export = etrade_df.groupby('year')['value'].sum()

plt.figure(figsize=(15, 6))

ax = sns.lineplot(x=yearly_export.index,y=yearly_export)

ax.set_title('Total exporting in the last 8 years')
yearly_import = itrade_df.groupby('year')['value'].sum()

plt.figure(figsize=(15, 6))

ax = sns.lineplot(x=yearly_import.index,y=yearly_import)

ax.set_title('Total importing in the last 8 years')
importer_lists = []

for i in etrade_df['year'].unique():

    importer_lists.extend(etrade_df[etrade_df['year'] == i][['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).iloc[0:3,:].index)
exporter_lists = []

for i in itrade_df['year'].unique():

    exporter_lists.extend(itrade_df[itrade_df['year'] == i][['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).iloc[0:3,:].index)
from collections import Counter

favor_importer = Counter(importer_lists).most_common(3)



plt.figure(figsize=(12, 5))

for country, count in favor_importer:

    importer = etrade_df[etrade_df['country'] == country][['year','value','country']].groupby(['year']).sum()

    ax = sns.lineplot(x= importer.index, y= importer['value'])

ax.set_title('Top 3 favourite importers')
favor_exporter = Counter(exporter_lists).most_common(3)



plt.figure(figsize=(12, 5))

for country, count in favor_exporter:

    exporter = itrade_df[itrade_df['country'] == country][['year','value','country']].groupby(['year']).sum()

    ax = sns.lineplot(x= exporter.index, y= exporter['value'])

ax.set_title('Top 3 favourite exporters')
trade_partners = etrade_df[['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).head()
plt.figure(figsize=(15, 6))

ax = sns.barplot(trade_partners.index, trade_partners.value, palette='Blues_d')

ax.set_title('Top 5 exporting partners')
trade_commodities = etrade_df[['Commodity','value']].groupby(['Commodity']).sum().sort_values(by = 'value', ascending = False).head(10)

trade_commodities
ax = sns.barplot(trade_commodities.value,trade_commodities.index, palette='Greens_d')

ax.set_title('Top 10 exporting commodities')
exporting_products = []

for i in etrade_df['year'].unique():

    exporting_products.extend(etrade_df[etrade_df['year'] == i][['Commodity','value']].groupby(['Commodity']).sum().sort_values(by = 'value', ascending = False).iloc[0:3,:].index)
from collections import Counter

favor_products = Counter(exporting_products).most_common(3)



plt.figure(figsize=(12, 5))

for product, count in favor_products:

    products = etrade_df[etrade_df['Commodity'] == product][['year','value','country']].groupby(['year']).sum()

    ax = sns.lineplot(x= products.index, y= products['value'])

    print(product)

ax.set_title('Top 3 favourite products')