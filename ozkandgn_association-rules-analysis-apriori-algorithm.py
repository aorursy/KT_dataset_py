import pandas as pd
### For apriori algorithm



!pip install mlxtend
### Groceries dataset



data = pd.read_csv("../input/groceries/groceries.csv",error_bad_lines=False,header=None);
### Count of the values



data.apply(data.value_counts).sum(axis=1).sort_values(ascending=False)
### Pandas Dataframe to list



### We need product lists for apriori algorithm



records = []

for i in range((len(data))):

    records.append([str(data.values[i,j]) for j in range(0, data.shape[1])])

records[:10]
### True - False (Dummy) array



from mlxtend.preprocessing.transactionencoder import TransactionEncoder



te = TransactionEncoder()

te_data = te.fit(records).transform(records)

df = pd.DataFrame(te_data, columns = te.columns_)

df = df.drop(("nan"),axis=1)

df
### Apriori algorithm

### Coexistence rate of products





from mlxtend.frequent_patterns import apriori



df1 = apriori(df, min_support=0.003, use_colnames = True, verbose=1)

df1
from mlxtend.frequent_patterns import association_rules



rules = association_rules(df1, metric = "confidence", min_threshold = 0.2)

rules
rules[rules["lift"]>2]