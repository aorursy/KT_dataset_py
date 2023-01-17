import pandas as pd

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules
import os

os.listdir('../input/online-retail-data-set-from-uci-ml-repo')
df = pd.read_excel('../input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')

df.head()
#some of the descriptions have spaces that need to be removed. 

#We’ll also drop the rows that don’t have invoice numbers and remove the credit transactions 

#(those with invoice numbers containing C).

df['Description'] = df['Description'].str.strip()

df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

df['InvoiceNo'] = df['InvoiceNo'].astype('str')

df = df[~df['InvoiceNo'].str.contains('C')]

df.head()
#Sales for France (to keep the data small)



#For each invoice , the count of each item is calculated,





basket = (df[df['Country'] =="France"]

          .groupby(['InvoiceNo', 'Description'])['Quantity']

          .sum().unstack().reset_index().fillna(0)

          .set_index('InvoiceNo'))

basket.head()
#any positive values are converted to a 1 and anything less the 0 is set to 0.

def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1



basket_sets = basket.applymap(encode_units)

basket_sets.drop('POSTAGE', inplace=True, axis=1)

basket_sets.head()
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

frequent_itemsets.head()
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules.head()
rules[ (rules['lift'] >= 6) &

       (rules['confidence'] >= 0.8) ]