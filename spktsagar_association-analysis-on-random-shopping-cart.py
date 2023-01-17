import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# mlxtend library
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
raw_data = pd.read_csv('../input/dataset.csv', header=None, sep='^')
raw_data.head(3)
max_col_count = max([row.count(',') for row in raw_data[0]])
raw_data = pd.read_csv('../input/dataset.csv', header=None, names=list(range(max_col_count)))
for col in raw_data.columns:
    raw_data[col] = raw_data[col].str.strip()
raw_data.head(3)
def strip_date_text(s):
    match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})([a-z /-]+)', s)
    date = datetime.strptime(match.groups()[0], '%d/%m/%Y').date()
    return date, match.groups()[1]
transac_data = pd.concat([pd.DataFrame(raw_data[0].apply(strip_date_text).tolist()), raw_data.iloc[:, 1:]], axis=1, ignore_index=True)
transac_data.head(3)
dataset = [transac_data.loc[i].dropna()[1:-1].values for i in range(len(transac_data))]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.head()
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules
min_supports = np.linspace(0.1, 1, 50, endpoint=True)
no_of_frequent_items = []

for min_support in min_supports:
    no_of_frequent_items.append(apriori(df, min_support=min_support).shape[0])
    
plt.plot(min_supports, no_of_frequent_items)
plt.xlabel('min_support')
plt.ylabel('number of frequent itemsets')
plt.title('min_support vs number of frequent itemsets')
min_supports = np.linspace(0.1, 1, 10, endpoint=True)
no_of_frequent_items = {}

plt.figure()
ax = plt.gca()

for min_support in min_supports:
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    itemset_sizewise_count = frequent_itemsets.groupby(['length']).size().tolist()
    print(itemset_sizewise_count)
    ax.plot(list(range(1, len(itemset_sizewise_count) + 1)), itemset_sizewise_count,
            label='min_support=' + str(min_support))
    
plt.legend()
plt.xlabel('size of itemsets')
plt.ylabel('number of frequent itemsets')