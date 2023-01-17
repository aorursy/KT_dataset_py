# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.frequent_patterns import apriori, association_rules

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
groceryData = pd.read_csv('../input/supermarket/GroceryStoreDataSet.csv',names=['Products'],header=None)
groceryData.head()
groceryData.info()
items=[]
for i in groceryData.values:
    items.extend( i[0].split(",")  )
items=list(set(items))
items
df=pd.DataFrame(data=0,columns=items,index=range(len(groceryData)))
for i in df.columns:
    df[i] = groceryData['Products'].str.contains(i)
df
df_freq = apriori(df, min_support = 0.1, use_colnames = True)
df_freq
association_rules(df_freq, metric = "lift", min_threshold = 1).sort_values(by=['antecedent support','confidence'],ascending=False).reset_index(drop=True).head(20)
df_freq['item_count'] = df_freq['itemsets'].apply(lambda x:len(x))
df_freq[(df_freq['item_count']==2) & (df_freq['support']>0.1)].sort_values(by='support',ascending=False)

