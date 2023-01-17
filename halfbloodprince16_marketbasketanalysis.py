import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/breadbasketanalysis/BreadBasket_DMS.csv")
df.info()
df.head()
df['Item'].unique()
df.drop(df[df['Item']=='NONE'].index, inplace=True)
# Year

df['Year'] = df['Date'].apply(lambda x: x.split("-")[0])

# Month

df['Month'] = df['Date'].apply(lambda x: x.split("-")[1])

# Day

df['Day'] = df['Date'].apply(lambda x: x.split("-")[2])
#top 15 most sold items

ms = df['Item'].value_counts().head(15)

print(ms)
#Monthly transactions

df.groupby('Month')['Transaction'].nunique().plot(kind='bar', title='Monthly Sales')

plt.show()
from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import association_rules, apriori
unq_transactions = []



for i in df['Transaction'].unique():

    tlist = list(set(df[df['Transaction']==i]['Item']))

    if len(tlist)>0:

        unq_transactions.append(tlist)

print(len(unq_transactions))
te = TransactionEncoder()

te_ary = te.fit(unq_transactions).transform(unq_transactions)

df2 = pd.DataFrame(te_ary, columns=te.columns_)
df2.head()
frequent_itemsets = apriori(df2, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

rules.sort_values('confidence', ascending=False)