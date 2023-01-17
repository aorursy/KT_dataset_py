import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/deliveries.csv')

df.head()
df.columns
df['over'].unique()
df.over.plot(kind='hist')

plt.show()
df.loc[10:20, 'over'] = 22
df.over.plot(kind='hist', bins=40)

plt.show()
df['over'].value_counts()
df['over'].describe()
import numpy as np
df.loc[df['over'] > 20, 'over'] = df.over.mean()
df.loc[10:20, 'over']
df.over.value_counts()
df.extra_runs.unique()
df.extra_runs.value_counts()
df[['total_runs', 'extra_runs']].plot(kind='hist', bins=10, alpha=0.5)

plt.show()
plt.figure(figsize=(10, 5))

df[(df['ball'] == 6) & (df['over'] < 15)]['total_runs'].plot('hist', normed=1, bins=20, align='left')

df[(df['ball'] == 6) & (df['over'] >= 15)]['total_runs'].plot('hist', normed=1, bins=20, align='right')

plt.legend(['over<15', 'over>=15'])

plt.show()
df[(df['ball'] == 6) & (df['over'] < 15)].shape
df[(df['ball'] == 6) & (df['over'] >= 15)].shape
below_thr_count = df[(df['ball'] == 6) & (df['over'] < 15)]['total_runs'].shape[0]

(df[(df['ball'] == 6) & (df['over'] < 15)]['total_runs'].value_counts()/below_thr_count).T
df['total_runs'].plot('density')

plt.show()
df[df['inning'] == 1].groupby(['match_id'])['over'].max().describe()
df[df['inning'] == 2].groupby(['match_id'])['over'].max().describe()
def cdf(x): 

    count = 0.0

    for value, value_count in sample.items():

        if value <= x:

            count += value_count

    prob = count/sum(list(sample.values()))

    return prob
df[df['inning'] == 1].groupby(['match_id'])['over'].max().value_counts()
sample = sample = df[df['inning'] == 1].groupby(['match_id'])['over'].max().value_counts().to_dict()

inn_1_cdf = df[df['inning'] == 1].groupby(['match_id'])['over'].max().value_counts()

inn_1_cdf = inn_1_cdf.reset_index()

inn_1_cdf.columns = ['over', 'count']

inn_1_cdf['cdf_inning_1'] = inn_1_cdf['over'].apply(cdf)

inn_1_cdf.set_index('over', inplace=True)

print(inn_1_cdf.head())

print('total matches', inn_1_cdf['count'].sum())

print('over 20 percentage', 545/577)
df[df['inning'] == 2].groupby(['match_id'])['over'].max().value_counts()
sample = sample = df[df['inning'] == 2].groupby(['match_id'])['over'].max().value_counts().to_dict()

inn_2_cdf = df[df['inning'] == 2].groupby(['match_id'])['over'].max().value_counts()

inn_2_cdf = inn_2_cdf.reset_index()

inn_2_cdf.columns = ['over', 'count']

inn_2_cdf['cdf_inning_2'] = inn_2_cdf['over'].apply(cdf)

inn_2_cdf.set_index('over', inplace=True)

print(inn_2_cdf.head())

print('total matches', inn_2_cdf['count'].sum())

print('over 20 percentage', 333/575)
cdf_df = pd.concat([inn_1_cdf[['cdf_inning_1']], inn_2_cdf[['cdf_inning_2']]], axis=1)

cdf_df.fillna(0, inplace=True)

cdf_df
ax = cdf_df.plot(drawstyle='steps', subplots=False)

ax.set(xlabel='Maximum overs', ylabel='CDF')

plt.show()