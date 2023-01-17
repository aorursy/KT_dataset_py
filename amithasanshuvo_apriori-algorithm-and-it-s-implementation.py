import numpy as np 

import pandas as pd 

from mlxtend.frequent_patterns import apriori, association_rules
data = pd.read_csv('../input/supermarket/GroceryStoreDataSet.csv',names=['Products'],header=None)
data.head()
data.values
data = list(data["Products"].apply(lambda x:x.split(',')))

data 
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)

df.head()
frq_items = apriori(df, min_support = 0.1, use_colnames = True)
frq_items
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 

rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())
