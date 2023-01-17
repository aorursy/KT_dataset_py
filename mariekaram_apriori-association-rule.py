import pandas as pd
df = pd.read_csv('../input/GroceryStoreDataSet.csv',names=['products'],header=None)
df
df.columns 
df.values
data = list(df["products"].apply(lambda x:x.split(',')))
data 
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)
df
from mlxtend.frequent_patterns import apriori
df1 = apriori(df,min_support=0.01,use_colnames=True)
df1
df1.sort_values(by="support",ascending=False)
df1['length'] = df1['itemsets'].apply(lambda x:len(x))
df1
df1[(df1['length']==2) & (df1['support']>=0.05)]