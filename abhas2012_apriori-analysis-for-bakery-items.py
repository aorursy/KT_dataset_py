# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/BreadBasket_DMS.csv")
data.head()
data.count()
data['Transaction'].nunique()
data['Item'].nunique()
most_frequent_items=data['Item'].value_counts()[:11]
most_frequent_items=most_frequent_items[most_frequent_items.index!='NONE']
most_frequent_items
plt.figure(figsize=[10,10])
most_frequent_items.plot(kind='bar')
plt.title('Most Frequently Bought Items')
plt.xlabel('Items')
plt.ylabel('Count')
plt.show()
data['Amount']=1
items=data.groupby(['Transaction','Item'])['Amount'].sum().unstack().fillna(0)
items.head().shape
def encode(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
itemsset=items.applymap(encode)    

from mlxtend.frequent_patterns import apriori  
from mlxtend.frequent_patterns import association_rules
frequent_items = apriori(itemsset, min_support=0.04, use_colnames=True)
print(frequent_items)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.head()

