import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/apple-store-ranks-2019/ranks.csv', index_col=['date'], parse_dates=['date'])

df.head()
df.info()
df.shape
free_5000 = df[(df['category'] == 5000) & (df['feed'] == 'free')]

free_5000.head()
free_6024 = df[(df['category'] == 6024) & (df['feed'] == 'free')]

free_6024.head()
f = free_5000.corr()

sns.heatmap(f, annot=True)
f = free_6024.corr()

sns.heatmap(f, annot=True)
# 菜鸟裹裹

free_5000[free_5000.appid==951610982]['ranking'].plot(legend=True)
free_5000[free_5000.appid==951610982]['change'].plot(legend=True)
free_5000['daily-return'] = free_5000['ranking'].pct_change()

free_5000['daily-return'].plot(legend=True, figsize=(18, 10), linestyle='--', marker='o')

plt.show()
sns.distplot(free_5000['daily-return'].dropna(), bins=100, color='purple')

plt.show()
y = free_5000[free_5000['appid']== 1460082863]['ranking'].pct_change()

y
z = free_5000[free_5000['ranking'] <= 200]

a = set(z.loc['2019-11-30':'2019-12-01']['appid'])

b = set(z.loc['2019-12-02']['appid'])

n = b - a

n