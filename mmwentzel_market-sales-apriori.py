# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import seaborn as sns

import datetime

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Import data

df = pd.read_csv('../input/market_data_02.csv')



# Valid Data

df.head() 

df.info()

df.columns
# Checking all unique items that are sold

print("Unique products: " + str(len(df['cod_prod'].unique())))
# Print the most sold items by transaction

fig, ax=plt.subplots(figsize=(16,7))

df['descricao'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1)

plt.xlabel('Product Item',fontsize=20)

plt.ylabel('Number of transactions',fontsize=17)

ax.tick_params(labelsize=20)

plt.title('20 Most Sold Items',fontsize=20)

plt.grid()

plt.ioff()
# Add columns of time, to get more insights

df['datetime'] = pd.to_datetime(df['data_emissao']+" "+df['hora_emissao'])

df['Week'] = df['datetime'].dt.week

df['Month'] = df['datetime'].dt.month

df['Weekday'] = df['datetime'].dt.weekday

df['Hours'] = df['datetime'].dt.hour
# Import the libraries to apriori

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules



# Group by before to fit

hot_encoded_df = df.groupby(['nota_fiscal_id', 'descricao'])['descricao'].count().unstack().reset_index().fillna(0).set_index('nota_fiscal_id')

hot_encoded_df.head()
def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1

hot_encoded_df = hot_encoded_df.applymap(encode_units)
# Fit the data 

frequent_itemsets = apriori(hot_encoded_df, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

rules.head(10)
# Get data and set to csv

rules.to_csv('market_data_out_all_results.csv')
# Filter confidence to the new dataset

dt3 = rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5)]



# Save data in csv

dt3.to_csv('market_data_out_filtered.csv')
support = rules.as_matrix(columns=['support'])

confidence = rules.as_matrix(columns=['confidence'])

import seaborn as sns



for i in range (len(support)):

    support[i] = support[i]

    confidence[i] = confidence[i]

    

plt.title('Assonciation Rules')

plt.xlabel('support')

plt.ylabel('confidance')

sns.regplot(x=support, y=confidence, fit_reg=False)