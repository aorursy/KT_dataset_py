import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
'''
Data From University of California, Irvine. Machine Learning Repository
'''
df = pd.read_csv('ApprioriData20180826.csv',encoding='latin-1')
df = df[0:100000]
df.head()
'''
Data Cleansing

'''
df['Description'] = df['Description'].str.strip() #Removing Space FROM Description Column
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True) #Drop Down Rows that don't have InvoiceNo
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
#df = df[~df['InvoiceNo'].str.contains('C')] # Removing Credit Card Transaction, Starting with C


print(df.head())
'''
For Simplicity, Taking Data Only For France
'''
basket = (df[df['Country'] =="Bangladesh"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))
print(df['Country'].size)
print(df['StockCode'].size)
print(df['Description'].size)
print(df['Quantity'].size)
'''
One Hot Encoding
'''
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
print(basket_sets.dropna())
#basket_sets.drop('POSTAGE', inplace=True, axis=1)
#basket_sets.isnull().any().any()
nan_rows = basket_sets[basket_sets.isnull().any(1)]
print(nan_rows)
basket_sets.fillna(0)
'''
Generating Frequent Item Sets that have a support of at least 7%
'''

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
import numpy as np
basket_sets.fillna(0)
basket_sets = basket_sets.replace(np.nan, 0)

basket_sets.loc[:, basket_sets.isna().any()]
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
print(type(rules))
with pd.option_context('display.max_rows', None, 'display.max_columns', 9):
    print(rules)

for i in range (len(rules)):
    #print(rules.values[i][1])
    print(list(rules.values[i][1])[0])

import csv
csv.register_dialect('myDialect',quoting=csv.QUOTE_ALL,skipinitialspace=True)

#print(len(rfmTable))
AprioriOut = [["ID","antecedents","consequents","antecedent support","consequent support","support","confidence","lift","leverage","conviction"]]    
for i in range (len(rules)):
    #print(str(rfmTable.values[i][0])+ '->'+str(rfmTable.values[i][1]) +'->' + str(rfmTable.values[i][2])+'->'+ str(kmeans.labels_[i]))
    row = []
    row.append(i+1)
    row.append(list(rules.values[i][0])[0])
    row.append(list(rules.values[i][1])[0])
    row.append(rules.values[i][2])
    row.append(rules.values[i][3])
    row.append(rules.values[i][4])
    row.append(rules.values[i][5])
    row.append(rules.values[i][6])
    row.append(rules.values[i][7])
    row.append(rules.values[i][7])
    #row.append(labels[i])
    AprioriOut.append(row)

with open('AprioriOut.csv', 'w',newline='',encoding='utf-8') as f:
    writer = csv.writer(f, dialect='myDialect')
    for row in AprioriOut:
        writer.writerow(row)

f.close()
