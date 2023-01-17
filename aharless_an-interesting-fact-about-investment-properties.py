import numpy as np 

import pandas as pd 
train = pd.read_csv('../input/train.csv')
train[train.price_doc.isin([1e6, 2e6, 3e6, 4e6, 5e6])].product_type.value_counts()
train[~train.price_doc.isin([1e6, 2e6, 3e6, 4e6, 5e6])].product_type.value_counts()
train[train.product_type=="Investment"].price_doc.value_counts().head(20)
train[~(train.product_type=="Investment")].price_doc.value_counts().head(20)
print( "\nAmong", train[(train.product_type=="Investment")].price_doc.count(), 

      "investment sales, there were only", 

      train[(train.product_type=="Investment")].price_doc.nunique(), "unique prices.\n")

print( "Among", train[~(train.product_type=="Investment")].price_doc.count(), 

      "owner-occupant sales, there were", 

      train[~(train.product_type=="Investment")].price_doc.nunique(), "unique prices." )