import pandas as pd
df = pd.read_csv('GroceryStoreDataSet.csv',names=['Products'], header=None)
df.columns
df.values
data = list(df["Products"].apply(lambda x:x.split(',')))
data
!pip install mlxtend
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)
df
from mlxtend.frequent_patterns import apriori
df1 = apriori(df,min_support=0.2,use_colnames=True)
df1
print("Kural_Sayısı:", len(apriori(df, min_support=0.2)))
df1.sort_values(by="support",ascending=False)
df1['length'] = df1['itemsets'].apply(lambda x:len(x))
df1[(df1['length']>=2) & (df1['support']>=0.2)]
from mlxtend.frequent_patterns import association_rules
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.40)
rules
