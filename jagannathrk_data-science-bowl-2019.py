import pandas as pd

pd.set_option('display.max_columns', None)

import numpy as np

import seaborn as sns

import matplotlib.style as style

style.use('fivethirtyeight')

import matplotlib.pylab as plt

import calendar

import warnings

warnings.filterwarnings("ignore")
%%time

train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_sub = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
train.head()
train.shape
keep_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()

train = pd.merge(train, keep_id, on="installation_id", how="inner")
train.shape
keep_id.shape
plt.rcParams.update({'font.size': 16})



fig = plt.figure(figsize=(12,10))

ax1 = fig.add_subplot(211)

ax1 = sns.countplot(y="type", data=train, color="blue", order = train.type.value_counts().index)

plt.title("number of events by type")



ax2 = fig.add_subplot(212)

ax2 = sns.countplot(y="world", data=train, color="blue", order = train.world.value_counts().index)

plt.title("number of events by world")



plt.tight_layout(pad=0)

plt.show()
def get_time(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    return df

    

train = get_time(train)



#list(set(train['title'].unique()).union(set(test['title'].unique())))
fig = plt.figure(figsize=(12,10))

se = train.groupby('date')['date'].count()

se.plot()

plt.title("Event counts by date")

plt.xticks(rotation=90)

plt.show()
fig = plt.figure(figsize=(12,10))

se = train.groupby('dayofweek')['dayofweek'].count()

se.index = list(calendar.day_abbr)

se.plot.bar()

plt.title("Event counts by day of week")

plt.xticks(rotation=0)

plt.show()
fig = plt.figure(figsize=(12,10))

se = train.groupby('hour')['hour'].count()

#se.index = list(calendar.day_abbr)

se.plot.bar()

plt.title("Event counts by hour of day")

plt.xticks(rotation=0)

plt.show()
test.head()
test.shape
test.installation_id.nunique()
sample_sub.shape
plt.rcParams.update({'font.size': 22})



plt.figure(figsize=(12,6))

sns.countplot(y="title", data=train_labels, color="blue", order = train_labels.title.value_counts().index)

plt.title("Counts of titles")

plt.show()
plt.rcParams.update({'font.size': 16})



plt.figure(figsize=(8,8))

sns.countplot(x="accuracy_group", data=train_labels, color="blue")

plt.title("Counts of accuracy group")

plt.show()
train_labels[train_labels.installation_id == "0006a69f"]
train[(train.event_code == 4100) & (train.installation_id == "0006a69f") & (train.title == "Bird Measurer (Assessment)")]
#credits for this code chuck go to Andrew Lukyanenko

train['attempt'] = 0

train.loc[(train['title'] == 'Bird Measurer (Assessment)') & (train['event_code'] == 4110),\

       'attempt'] = 1

train.loc[(train['type'] == 'Assessment') &\

       (train['title'] != 'Bird Measurer (Assessment)')\

       & (train['event_code'] == 4100),\

          'attempt'] = 1



train['correct'] = None

train.loc[(train['attempt'] == 1) & (train['event_data'].str.contains('"correct":true')), 'correct'] = True

train.loc[(train['attempt'] == 1) & (train['event_data'].str.contains('"correct":false')), 'correct'] = False
train[(train.installation_id == "0006a69f") & (train.attempt == 1)]
train[~train.installation_id.isin(train_labels.installation_id.unique())].installation_id.nunique()
train = train[train.installation_id.isin(train_labels.installation_id.unique())]

train.shape
print(f'Number of rows in train_labels: {train_labels.shape[0]}')

print(f'Number of unique game_sessions in train_labels: {train_labels.game_session.nunique()}')
count_combi = train.groupby(['game_session', 'world']).size()

print(f'Number of unique game_session in train: {train.game_session.nunique()}')

print(f'Number of unique combi of game_session and world in train: {count_combi.shape[0]}')
train_labels = pd.merge(train_labels, train[['game_session', "world"]].drop_duplicates(), on= "game_session", how="left")
train_labels.shape
train_labels.head()