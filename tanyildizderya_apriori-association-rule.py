import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/supermarket/GroceryStoreDataSet.csv',names=['products'],header=None)
df.head()
df.columns 
df.values
data = list(df["products"].apply(lambda x:x.split(',')))

data 
from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)

df.head()
dff = apriori(df,min_support=0.01,use_colnames=True)

dff
dff['length'] = dff['itemsets'].apply(lambda x:len(x))

dff
dff.describe().T
dff[(dff['length']==2) & (dff['support']>=0.05)]
dff[(dff['length']==3) & (dff['support']>=0.05)]
dff[(dff['length']==4) & (dff['support']>=0.05)]