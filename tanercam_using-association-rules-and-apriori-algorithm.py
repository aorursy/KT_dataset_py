import pandas as pd
import numpy as np
!pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv('../input/supermarket/GroceryStoreDataSet.csv',
                 names = ['products'], header = None)
df
df.columns
df.values
data = list(df['products'].apply(lambda x:x.split(',')))
data
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data, columns = te.columns_)
df
from mlxtend.frequent_patterns import apriori
df1 = apriori(df, min_support = 0.2, use_colnames=True, verbose =1)
df1
association_rules(df1, metric = 'confidence', min_threshold = 0.6)