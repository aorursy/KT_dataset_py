import pandas as pd

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('../input/GroceryStoreDataSet.csv',names=['products'],header=None)

df
data = list(df["products"].apply(lambda x:x.split(',')))

te = TransactionEncoder()

te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data,columns=te.columns_)

df 
df1 = apriori(df,min_support=0.01,use_colnames=True)

df1.sort_values(by="support",ascending=False)
assos = association_rules(df1, metric = "confidence", min_threshold = 0.3)

assos[assos['lift']>1]