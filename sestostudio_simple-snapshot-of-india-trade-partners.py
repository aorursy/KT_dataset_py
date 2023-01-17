# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
trade_df = pd.read_csv('../input/india-trade-data/2018-2010_export.csv')

trade_df.head()
trade_df.info()
trade_df['value'].isnull().sum()
trade_df["value"].fillna(trade_df.groupby('Commodity')['value'].transform('mean'),inplace = True)
trade_df['value'].isnull().sum()
yearly_export = trade_df.groupby('year')['value'].sum()

plt.figure(figsize=(15, 6))

ax = sns.lineplot(x=yearly_export.index,y=yearly_export)

ax.set_title('Total exporting in the last 8 years')
importer_lists = []

for i in trade_df['year'].unique():

    importer_lists.extend(trade_df[trade_df['year'] == i][['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).iloc[0:3,:].index)
from collections import Counter

favor_importer = Counter(importer_lists).most_common(3)



plt.figure(figsize=(12, 5))

for country, count in favor_importer:

    importer = trade_df[trade_df['country'] == country][['year','value','country']].groupby(['year']).sum()

    ax = sns.lineplot(x= importer.index, y= importer['value'])

ax.set_title('Top 3 favourite importers')
trade_partners = trade_df[['country','value']].groupby(['country']).sum().sort_values(by = 'value', ascending = False).head()
plt.figure(figsize=(15, 6))

ax = sns.barplot(trade_partners.index, trade_partners.value, palette='Blues_d')

ax.set_title('Top 5 exporting partners')
trade_commodities = trade_df[['Commodity','value']].groupby(['Commodity']).sum().sort_values(by = 'value', ascending = False).head()

trade_commodities
ax = sns.barplot(trade_commodities.value,trade_commodities.index, palette='Reds_d')

ax.set_title('Top 5 exporting commodities')
exporting_products = []

for i in trade_df['year'].unique():

    exporting_products.extend(trade_df[trade_df['year'] == i][['Commodity','value']].groupby(['Commodity']).sum().sort_values(by = 'value', ascending = False).iloc[0:3,:].index)
from collections import Counter

favor_products = Counter(exporting_products).most_common(3)



plt.figure(figsize=(12, 5))

for product, count in favor_products:

    products = trade_df[trade_df['Commodity'] == product][['year','value','country']].groupby(['year']).sum()

    ax = sns.lineplot(x= products.index, y= products['value'])

    print(product)

ax.set_title('Top 3 favourite products')