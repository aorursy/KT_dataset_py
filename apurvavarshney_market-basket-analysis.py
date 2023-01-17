# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Read the dataset
df = pd.read_csv('/kaggle/input/online-retail-data-set-from-ml-repository/retail_dataset.csv', sep=',')
df.head()
df.info()
## Make list of the dataset
records = []
for i in range(1, 315):
    records.append([str(df.values[i, j]) for j in range(0, 7)])
records
## Encode the data for machine to read 

te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
df1 = pd.DataFrame(te_ary, columns = te.columns_)
df1.head()
## Find Frequent items using apriori algorithm

frequent_itemsets = apriori(df1, min_support=0.2, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
## association with confidence metric

association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules
## Find the items which match the required criteria

rules[ (rules['antecedent_len'] >= 2) &
       (rules['confidence'] > 0.5) &
       (rules['lift'] > 1.0) ]
## Plot a graph between lift and confidence values using antecedent length

import matplotlib.pyplot as plt
from mlxtend.plotting import category_scatter

fix = category_scatter(x = "lift", y = "confidence", label_col = "antecedent_len", 
                       data=rules, legend_loc= "lower right")
