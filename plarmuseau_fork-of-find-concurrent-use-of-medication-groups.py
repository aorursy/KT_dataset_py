import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/vb01.csv' ,names=['atcnr','atc','doping','ddd','dpp','ehd','tva','nr','prdnr','code','aant','pp','salesnr','client','date'] )

df
df['atc3']=df['atc'].str[:3]
df['atc5']=df['atc'].str[:5]
df['tel']=1.0
df
basket_sets = pd.pivot_table(df, index='salesnr',columns='atc5',values='tel')
basket_sets
from mlxtend.frequent_patterns import apriori # Data pattern exploration
from mlxtend.frequent_patterns import association_rules # Association rules conversion
from mlxtend.preprocessing import OnehotTransactions # Transforming dataframe for apriori

#Apriori aplication: frequent_itemsets
# Note that min_support parameter was set to a very low value, this is the Spurious limitation, more on conclusion section
frequent_itemsets = apriori(basket_sets.fillna(0), min_support=0.000251, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Advanced and strategical data frequent set selection
frequent_itemsets[ (frequent_itemsets['length'] > 1) &
                   (frequent_itemsets['support'] >= 0.02) ].head()
# Generating the association_rules: rules
# Selecting the important parameters for analysis
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('support', ascending=False).head()
rules.plot.scatter(x='support',y='confidence',c='lift', colormap='YlOrRd')
rules.sort_values('lift', ascending=False)