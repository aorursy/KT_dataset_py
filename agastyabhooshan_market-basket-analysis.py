import pandas as pd

import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv('../input/groceries-dataset/Groceries_dataset.csv', parse_dates=[1]) 
df.dtypes
df.isna().sum()
df
# First, get a list of all items. Easiest way to do this is by dummyfying the item description column

items_dummies = pd.get_dummies(df['itemDescription'])
# Baskets will be a list of items bought together

baskets = df.groupby(['Date', 'Member_number']).agg(lambda x: ','.join(x).split(','))['itemDescription'].values
baskets, len(baskets)
df = df.join(items_dummies)
df
df.drop('itemDescription', axis=1, inplace=True)
df = df.groupby(['Date', 'Member_number']).sum() 
df.head()
df['basket'] = baskets

df.head(1) # Added the basket column to the dataset
# Check if the calculation is ok

(df.sum(axis=1) != df['basket'].apply(len)).sum() # Perfect
# There are samples where there are more than one of the same item in the basket (eg. {milk, milk})

# We need to only keep 1

len(np.where(df.drop('basket', axis=1)>1)[0])
for i in df.drop('basket', axis=1):

    df[i] = df[i].map(lambda x: 1 if x >1 else x)
df
len(np.where(df.drop('basket', axis=1)>1)[0])
df['UHT-milk'].sum()/len(df) # An example of calculating support for an item
# Drop the item list from the dataframe. No longer needed, since we have verified that the encoding is correct.

df.drop('basket', axis=1, inplace=True)
# Lets keep the support threshold at 0.1%

supports = apriori(df, min_support=1e-3, use_colnames=True)
# Down to 69 items. How many associations can we have?

supports
# Number of itemsets which have over 1 item in the basket

supports[supports['itemsets'].map(lambda x: len(x)>1)] 
associations = association_rules(supports, metric='lift', min_threshold=1)
# Antecendent support is support for the antecedent, i.e. before adding the new item

# Consequent support is the support of the consequent, i.e. the new item (in the row 127, the consequent support

# is the same as the support of sausage)

associations = associations.sort_values('confidence', ascending=False)

associations.head(10)
associations.sort_values('lift', ascending=False).head(10)