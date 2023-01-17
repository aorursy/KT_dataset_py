!pip install pandas ml_extend
import numpy as np

import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules
data = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')

data = data.head(30000) #using only half data due to memory issues

data.head()
data.shape
#Append Quantity column. Since, there's 1 product on every row we can easily use Quantity as 1 which will be added subsequently.

data.insert(7, 'quantity',1)

data.shape
data.head()
#Total no. of unique orders and products

data.describe(include = 'all')
data['product_id'].value_counts()
#If you want to cross check the data you can replace 10 with 80.

item_freq = data['product_id'].value_counts()

data = data[data.isin(item_freq.index[item_freq >= 10]).values]

data['product_id'].value_counts()
#Average products purchased per transaction

data['order_id'].value_counts().mean()
#Create a basket of all products, orders and quantity

basket = (data.groupby(['order_id','product_id'])['quantity']).sum().unstack().reset_index().fillna(0).set_index('order_id')

basket.head()
#Convert 0.0 to 0, convert the units to One hot encoded values

def encode_units(x):

    if x<= 0:

        return 0

    if x>=1:

        return 1

    

basket_sets = basket.applymap(encode_units)

basket_sets.head()
#Build frequent itemsets

frequent_itemsets = apriori(basket_sets, min_support = 0.0001, use_colnames = True)

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

frequent_itemsets
#Create Rules

rules = association_rules(frequent_itemsets, metric = 'support', min_threshold = 0.0001)

rules
#Products having 50% confidence likely to be purchased together

rules[rules['confidence'] >= 0.50]