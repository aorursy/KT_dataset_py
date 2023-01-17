import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules
# Load the dataset

ORD = pd.read_excel("../input/online-retail-ii-dataset/online_retail_II.xlsx")
# Display first 5 rows/transactions

ORD.head()
# Summary stats

ORD.info()
# checking for duplicate transactions

ORD.duplicated().sum()
print("Number of transactions before duplicates removal : %d " % ORD.shape[0])

# Dropping the duplicated transactions

ORD = ORD.drop(index=ORD[ORD.duplicated()].index)

print("Number of transactions after duplicates removal  : %d " % ORD.shape[0])
# Checking for cancelled transactions

ORD[ORD['Invoice'].astype(str).str[0] == 'C'].tail()
print("Number of transactions before dropping the cancelled transactions : %d " % ORD.shape[0])

# Dropping the cancelled transactions

ORD = ORD.drop(index=ORD[ORD['Invoice'].astype(str).str[0] == 'C'].index)

print("Number of transactions after dropping the cancelled transactions  : %d " % ORD.shape[0])
# Checking for missing values

ORD.isnull().sum()
# Remove transactions with missing product description

ORD = ORD.drop(index=ORD[ORD['Description'].isnull()].index)

# still any missing product descriptions ?

ORD.isnull().sum()
# Dropping transactions with negative quantity 

ORD = ORD.drop(index = ORD[ORD['Quantity'] <= 0].index)
# Summary stats for feature 'Country'

ORD['Country'].describe()
# transactions count by country

ORD['Country'].value_counts()
country = 'Germany'

ord_country = ORD[ORD['Country'] == country]
print("Number of unique invoices : %d " % len(ord_country['Invoice'].value_counts()))

print("Number of unique products : %d " % len(ord_country['Description'].value_counts()))
# Product sold quantity per invoice

freq = ord_country.groupby(['Invoice', 'Description'])['Quantity'].sum()
prod_freq = freq.unstack().fillna(0).reset_index().set_index('Invoice')

prod_freq.head()
# Set value to 1 for postivie quantity. Anything else set to 0

product_set = prod_freq.applymap(lambda x : 1 if x > 0 else 0 )

product_set.head()
# return the products and productsets with at least 10% support

frequent_products = apriori(product_set, min_support=0.1, use_colnames=True)

frequent_products['length'] = frequent_products['itemsets'].apply(lambda x : len(x))

# productset of length 2 

frequent_products[frequent_products.length > 1]
# Identiy frequent productsets with the level of confidence above the 70 percent threshold

rules = association_rules(frequent_products, metric="confidence", min_threshold=0.7)

rules
# Identify productsets with lift socre of >= 2

rules[rules['lift'] >= 2]