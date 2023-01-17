import pandas as pd
import numpy as np

%matplotlib inline
from matplotlib import pyplot as plt

import seaborn as sns
sns.set()

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

from scipy import stats
df = pd.read_csv('../input/magicpin_assignment.csv', encoding='latin-1')
df.head()
df.describe()
df.info()
df[df.isnull().any(axis=1)]
f, ax = plt.subplots(2, 1, figsize=(16, 10))
sns.distplot(df['bill size'], hist=False, ax=ax[0])
sns.distplot(df['cashback'].dropna(), hist=False, ax=ax[1])
f, ax = plt.subplots(2, 1, figsize=(16, 10))
sns.boxplot(df['bill size'], ax=ax[0])
sns.boxplot(df['cashback'].dropna(), ax=ax[1])
df.dropna(inplace=True)
zscore = stats.zscore(df[['bill size', 'cashback']])
# https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-dataframe
df_clipped = df[(np.abs(zscore) < 3).all(axis=1)]
f, ax = plt.subplots(2, 1, figsize=(16, 10))
sns.distplot(df_clipped['bill size'], hist=False, ax=ax[0])
sns.distplot(df_clipped['cashback'].dropna(), hist=False, ax=ax[1])
f, ax = plt.subplots(2, 1, figsize=(16, 10))
sns.boxplot(df_clipped['bill size'], ax=ax[0])
sns.boxplot(df_clipped['cashback'].dropna(), ax=ax[1])
merchant_count_df = pd.DataFrame(df_clipped['merchant_name'].value_counts()).reset_index()
merchant_count_df
f, ax = plt.subplots(figsize=(16, 10))
sns.barplot(x='index', y='merchant_name', data=merchant_count_df[:50])
plt.xticks(rotation=45, ha='right')
ax.set_xlabel('Merchant Name')
ax.set_ylabel('Count')
ax.set_title('Most popular merchants (Top 50)')
df_top_10 = pd.merge(merchant_count_df[:10], df_clipped, left_on='index', right_on='merchant_name', how='left')
df_top_10.drop(columns=['merchant_name_x', 'index'], inplace=True)
df_top_10.rename(index=str, columns={'merchant_name_y': 'merchant_name'}, inplace=True)
f, ax = plt.subplots(figsize=(16, 10))
sns.boxplot(x='merchant_name', y='cashback', data=df_top_10, hue='first transaction?', ax=ax)
f, ax = plt.subplots(figsize=(16, 10))
sns.boxplot(x='merchant_name', y='cashback', data=df_top_10, hue='funding trxn?', ax=ax)
sns.pairplot(df_clipped.loc[:, ['first transaction?', 'funding trxn?', 'bill size', 'cashback', 'followers']])
df_clipped[(df_clipped['first transaction?']==1) & (df_clipped['followers'] > 0)].shape[0]
df_clipped[df_clipped['bill size'] >= 5000].sort_values(['bill size', 'cashback'], ascending=False)
set_of_first_customers = df_clipped[df_clipped['first transaction?']==1]['customer_id']

print(set_of_first_customers.unique().shape[0])
print(set_of_first_customers.shape[0])
set_of_first_customers[set_of_first_customers.duplicated()]
df_clipped[df_clipped['customer_id'] == 1102010]
df_clipped[df_clipped['customer_id'] == 1101987]
df_clipped[df_clipped['customer_id'] == 1149507]
# Assumption
# Returning customers would also have a record where first transaction = 1
set_of_returning_customers = df_clipped[df_clipped['first transaction?'] == 0]['customer_id']
regular_or_returning_customers = set(set_of_returning_customers) - set(set_of_first_customers)
len(regular_or_returning_customers)
lost_customers = set(set_of_first_customers) - set(set_of_returning_customers)
len(lost_customers)
df_by_date = df_clipped.groupby('date', as_index=False)['bill size'].sum()
f, ax = plt.subplots(figsize=(16, 10))
plt.xticks(rotation=45, ha='right')
sns.lineplot(x='date', y='bill size', data=df_by_date, ax=ax)
df_clipped['mod_time'] = df_clipped['time'].apply(lambda x: ':'.join(x.split(':')[:-1]))
df_by_minute = df_clipped.groupby('mod_time', as_index=False)['bill size'].sum()
df_by_minute['mod_time'] = pd.to_datetime(df_by_minute['mod_time'], format='%H:%M').dt.time
f, ax = plt.subplots(figsize=(16, 10))
plt.xticks(rotation=90, ha='right')
sns.lineplot(x='mod_time', y='bill size', data=df_by_minute, ax=ax)
df_active_customers = pd.DataFrame(df_clipped['customer_id'].value_counts()).reset_index().sort_values('customer_id', ascending=False)
df_active_customers
f, ax = plt.subplots(figsize=(16, 10))
sns.barplot(x='index', y='customer_id', data=df_active_customers[:50])
plt.xticks(rotation=90)
ax.set_xlabel('Customer ID')
ax.set_ylabel('Trxns')
plt.title('50 most active customers')
df_clipped[df_clipped['customer_id'] == 220524]
df_clipped[df_clipped['customer_id'] == 1130580]