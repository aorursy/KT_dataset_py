import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules
PATH_INPUT = "/kaggle/input/"

PATH_WORKING = "/kaggle/working/"

PATH_TMP = "/tmp/"
df_raw = pd.read_csv(f'{PATH_INPUT}data.csv', encoding='iso-8859-1')
df_raw.shape
!ls -lh {PATH_INPUT}
df_raw.describe(include='all')
df = df_raw.copy()
df.query('Quantity < -80000')
df.query('Quantity > 80000')
basket = df.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack()

basket.shape
basket = basket.applymap(lambda x: 1 if x>0 else 0)
basket.head(1)
basket.iloc[0].value_counts()
%%time

itemsets = apriori(basket, min_support=0.005, use_colnames=True)
itemsets.shape
itemsets.sort_values('support',ascending=False).head()
rules = association_rules(itemsets, metric="lift", min_threshold=1)
rules.shape
sns.scatterplot(x='support', y='confidence', hue='lift', data=rules)

plt.show()
sns.scatterplot(x='support', y='confidence', hue='leverage', data=rules)

plt.show()
rules.sort_values('lift', ascending=False).head(10)