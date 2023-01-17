import os

import re

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import mlxtend

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

sns.set()
print(os.listdir("../input"))
retail_data = pd.read_excel('../input/Online Retail.xlsx',sep='delimiter')
retail_data.head(30)
retail_data.info()
retail_data.describe()
len(retail_data)
retail_data['Description'].unique().shape
retail_data['CustomerID'].unique().shape
(retail_data['Country'].unique())
retail_data.columns
retail_data.loc[retail_data['Quantity'].argmax()]
retail_data['Description'] = retail_data['Description'].str.strip()
retail_data['Description']
basket = (retail_data[retail_data['Country'] =="France"]

          .groupby(['InvoiceNo', 'Description'])['Quantity']

          .sum().unstack().reset_index().fillna(0)

          .set_index('InvoiceNo'))
basket.head()
def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1

basket_sets = basket.applymap(encode_units)

basket_sets.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules