%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
STATIONS = ['S32', 'S33', 'S34']

train_date_part = pd.read_csv('../input/train_date.csv', nrows=10000)

date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)

date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])

date_cols = date_cols[date_cols['station'].isin(STATIONS)]

date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()

print(date_cols)

train_date = pd.read_csv('../input/train_date.csv', usecols=['Id'] + date_cols)

print(train_date.columns)

train_date.columns = ['Id'] + STATIONS

for station in STATIONS:

    train_date[station] = 1 * (train_date[station] >= 0)

response = pd.read_csv('../input/train_numeric.csv', usecols=['Id', 'Response'])

print(response.shape)

train = response.merge(train_date, how='left', on='Id')

# print(train.count())

train.head(3)

train.to_csv("tr_merge.csv",index=False)









train_date = pd.read_csv('../input/test_date.csv', usecols=['Id'] + date_cols)

print(train_date.columns)

train_date.columns = ['Id'] + STATIONS

for station in STATIONS:

    train_date[station] = 1 * (train_date[station] >= 0)

response = pd.read_csv('../input/test_numeric.csv', usecols=['Id'])

print(response.shape)

train = response.merge(train_date, how='left', on='Id')

# print(train.count())

train.head(3)

train.to_csv("ts_merge.csv",index=False)
train['cnt'] = 1

failure_rate = train.groupby(STATIONS).sum()[['Response', 'cnt']]

failure_rate['failure_rate'] = failure_rate['Response'] / failure_rate['cnt']

failure_rate = failure_rate[failure_rate['cnt'] > 1000]  # remove 

failure_rate.head(20)
failure_rate_pretty = failure_rate.reset_index()

failure_rate_pretty['group'] = ['-'.join([s if row[s] else '' for s in STATIONS]) \

                         for _, row in failure_rate_pretty.iterrows()]

fig=plt.figure(figsize=(10, 4))

sns.barplot(x='group', y="failure_rate", data=failure_rate_pretty, color='r', alpha=0.8)

plt.ylabel('failure rate')

for i, row in failure_rate_pretty.iterrows():

    plt.text(i, row['failure_rate']+0.01, np.round(row['failure_rate'], 3),

             verticalalignment='top', horizontalalignment='center')

plt.title('Station combinations %s' % str(STATIONS))

fig.savefig('failure_rate.png', dpi=300)

plt.show()