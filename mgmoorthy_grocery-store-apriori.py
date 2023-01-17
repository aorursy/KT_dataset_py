#Learner stage

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori
#Read only products data from Kaggle

df = pd.read_csv('/kaggle/input/supermarket/GroceryStoreDataSet.csv',names=['products'],header=None)
#Check dataset 

df.head()
df.columns
df.tail()
df.values
#Split strings

data = list(df["products"].apply(lambda x:x.split(',')))

data
#Preprocessing

te = TransactionEncoder()

te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data,columns=te.columns_)

df.head()
df.columns
#Estimate demand of single product

df1 = apriori(df,min_support=0.01,use_colnames=True)

df1
df1.sort_values(by="support",ascending=False)
df1['length'] = df1['itemsets'].apply(lambda x:len(x))

df1
#confidence of products

df1[(df1['length']==3) & (df1['support']>=0.01)].sort_values(by='support',ascending=False)