import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.__version__

import os
print(os.listdir("../input"))
#We can see that the dataset has data of Data,Time,Transaction and the item sold at the bakery.
df = pd.read_csv('../input/BreadBasket_DMS.csv')

df.head(), df.info()
df['Item'] = df['Item'].str.lower()
x = df['Item'] == "none"
print(x.value_counts())
df = df.drop(df[df.Item == 'none'].index)
len(df['Item'].unique())
fig, ax=plt.subplots(figsize=(16,7))
df['Item'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=1)
plt.xlabel('Food Item',fontsize=20)
plt.ylabel('Number of transactions',fontsize=17)
ax.tick_params(labelsize=20)
plt.title('20 Most Sold Items at the Bakery',fontsize=20)
plt.grid()
plt.ioff()
df['datetime'] = pd.to_datetime(df['Date']+" "+df['Time'])
df['Week'] = df['datetime'].dt.week
df['Month'] = df['datetime'].dt.month
df['Weekday'] = df['datetime'].dt.weekday
df['Hours'] = df['datetime'].dt.hour

df1=df[['Date','Transaction', 'Month','Week', 'Weekday','Hours']]
df2['Counts'] = df1(['Date']).size().reset_index(name="counts")

sns.countplot(x='Weekday',data=df1)

sns.countplot(x='Hours',data=df1)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
hot_encoded_df = df.groupby(['Transaction', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
hot_encoded_df.head()
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
hot_encoded_df = hot_encoded_df.applymap(encode_units)
frequent_itemsets = apriori(hot_encoded_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
rules.head(10)
rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5)]
support = rules.as_matrix(columns=['support'])
confidence = rules.as_matrix(columns=['confidence'])
import seaborn as sns

for i in range (len(support)):
    support[i] = support[i]
    confidence[i] = confidence[i]
    
plt.title('Assonciation Rules')
plt.xlabel('support')
plt.ylabel('confidance')
sns.regplot(x=support, y=confidence, fit_reg=False)

