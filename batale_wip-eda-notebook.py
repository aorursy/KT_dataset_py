import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = (16,9)
import seaborn as sns
import scipy
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')
df.info()
df.head()
hist_kws={"alpha": 0.3}
sns.distplot(df.deal_probability, hist_kws=hist_kws)
plt.title('Deal Probability distribution')
plt.margins(0.02)
plt.show()
# We sample the dataframe & limit the extreme prices (+ no NaN !)
df_dropped = df.loc[df.index[df.price < 500000]].sample(n=50000)
sns.lmplot('price', 'deal_probability', hue='user_type', data=df_dropped, fit_reg=True, size=10, aspect=2, scatter_kws={'s': 10, 'alpha':0.3})
plt.xlabel('Ad price')
plt.ylabel('Deal Probability')
plt.margins(0.01)
plt.show()
print(f'Pearson correlation : {scipy.stats.pearsonr(df_dropped.price.values, df_dropped.deal_probability.values)}')
sns.distplot(df.price.dropna(), kde=False)
plt.margins(0.02)
plt.show()
sns.kdeplot(df_dropped.price)
plt.margins(0.02)
plt.show()
df_trimmed = df.loc[df.index[df.price < 40000]].sample(n=50000)
sns.kdeplot(df_trimmed.price)
plt.margins(0.02)
plt.show()
print('Counts:\n')
for user_type in df.user_type.unique():
    print(f"{user_type} users : {df[df.user_type == user_type].shape[0]}")
sns.violinplot('user_type', 'price', data=df_dropped)
sns.violinplot('user_type', 'price', data=df_trimmed)
hist_kws={"alpha": 0.2}
sns.distplot(df[df.user_type == 'Private']['deal_probability'], label='Private', hist_kws=hist_kws)
sns.distplot(df[df.user_type == 'Company']['deal_probability'], label='Company', hist_kws=hist_kws)
sns.distplot(df[df.user_type == 'Shop']['deal_probability'], label='Shop', hist_kws=hist_kws)
sns.distplot(df.deal_probability, label='All', hist_kws=hist_kws)
plt.title('Deal Probability distribution')
plt.legend()
plt.margins(0.02)
plt.show()
print('Counts\n')
print(f'region {len(df.region.unique())}')
print(f'city {len(df.city.unique())}')
print(f'parent cat {len(df.parent_category_name.unique())}')
print(f'cat {len(df.category_name.unique())}')
