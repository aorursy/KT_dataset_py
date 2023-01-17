import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
file_name = '../input/BreadBasket_DMS.csv'
bakery_df = pd.read_csv(file_name)
# Taking a Look at the Data
bakery_df.head()
pd.DataFrame(bakery_df.Item.value_counts()).sort_values('Item'
                                                        ,ascending=False)
bakery_df = bakery_df.drop(list(bakery_df[bakery_df['Item']=="NONE"].index), 
                           axis=0)
bakery_df.shape
print('Total number of Items sold at the bakery is:',bakery_df['Item'].nunique())
fig, ax=plt.subplots(figsize=(16,7))
bakery_df['Item'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Food Item',fontsize=20)
plt.ylabel('Number of transactions',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('20 Most Sold Items at the Bakery',fontsize=25)
plt.grid()
plt.ioff()
from mlxtend.frequent_patterns import apriori # Data pattern exploration
from mlxtend.frequent_patterns import association_rules # Association rules conversion
bakery_df_encoded=bakery_df.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
#Above line of code is transfrom data to make items as columns and each 
#transaction as a row and count same Items bought in one transaction but 
#fill other cloumns of the row with 0 to represent item which are not bought.
bakery_df_encoded.head()
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
bakery_df_encoded = bakery_df_encoded.applymap(encode_units)
frequent_itemsets = apriori(bakery_df_encoded, min_support=0.03, use_colnames=True)
frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules