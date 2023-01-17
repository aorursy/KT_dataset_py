%matplotlib inline

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/median_price.csv')

df = df.dropna(thresh=48)

df = df.drop('SizeRank', axis=1)
pivot = pd.pivot_table(df, index=['City','RegionName'])

pivot = pivot.T

pivot.ix['2016-01'].sort_values().plot(kind='barh', figsize=(5, 7), width=0.8, fontsize=10)

plt.grid(color='k', axis='x', alpha=0.4, lw=0.8)
plt.style.use('fivethirtyeight')

sns.set_style("whitegrid")



pivot[['Atlanta', 'Chicago', 'Dallas', 'Delray Beach', 'Fort Lauderdale', 'Hollywood',

       'Honolulu', 'Miami', 'Miami Beach', 'New York', 'Philadelphia', 'Pompano Beach',

       'Seattle']].plot(sharex = False, subplots=True,

        figsize=(8, 70), rot=45, fontsize=9)



plt.tight_layout()
cities = df[['City']].drop_duplicates().values.tolist()



for x in cities:

    pivot.ix['2016-01':][x].plot(figsize=(6, 4), fontsize=8)

    plt.legend(loc='upper center', fancybox=True, shadow=True, 

               bbox_to_anchor=(1.25, 1), fontsize=8)

    plt.tight_layout()