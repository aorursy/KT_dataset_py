# Presettings (imports and so on)

import numpy as np

import pandas as pd

import seaborn as sns

import pandas as pd



from sklearn import preprocessing

le = preprocessing.LabelEncoder()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sample_submission = pd.read_csv("../input/data-science-bowl-2019/sample_submission.csv")

specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

test = pd.read_csv("../input/data-science-bowl-2019/test.csv")

train = pd.read_csv("../input/data-science-bowl-2019/train.csv")

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
train_labels.info()
full_train = train.merge(train_labels, on='game_session', how='right')
train.head(5)
train.info()
train.describe()
sns.distplot(full_train['event_count']);
sns.relplot(x="event_count", y="accuracy_group", kind="line", data=full_train);
print('Correlation on full dataset:', round(full_train['accuracy_group'].corr(full_train['event_count']), 2))
sns.distplot(full_train['event_code']);
sns.relplot(x="event_code", y="accuracy_group", kind="line", data=full_train);
print('Correlation on full dataset:', round(full_train['accuracy_group'].corr(full_train['event_code']), 2))
sns.distplot(full_train['game_time']);
sns.boxplot(full_train['game_time']);
print('Correlation on full dataset:', round(full_train['accuracy_group'].corr(full_train['game_time']), 2))
print('Num of unique values in "Title" column:', full_train['title_x'].nunique())
ax = sns.barplot(x=full_train['title_x'], y=full_train['accuracy_group'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
full_train['title_x'] = le.fit_transform(full_train['title_x'])

print('Correlation on full dataset:', round(full_train['accuracy_group'].corr(full_train['title_x']), 2))
print('Num of unique values in "Type" column:', full_train['type'].nunique())
print('Num of unique values in "World" column:', full_train['world'].nunique())
ax = sns.barplot(x=full_train['world'], y=full_train['accuracy_group'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
train_with_specs = full_train.join(specs.set_index('event_id'), on='event_id')

print('Num of unique values in "info" column:', train_with_specs['info'].nunique())

print('Num of unique values in "args" column:', train_with_specs['args'].nunique())
train_with_specs.drop('info', axis=1, inplace=True)

train_with_specs['args'] = le.fit_transform(train_with_specs['args'])

train_with_specs.head(2)
sns.distplot(train_with_specs['args'])
ax = sns.barplot(x='args', y='accuracy_group', data=train_with_specs, color="salmon", saturation=.6)

p = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
print('Correlation on full dataset:', round(train_with_specs['accuracy_group'].corr(train_with_specs['args']), 2))
train_with_specs.drop('event_id', axis=1, inplace=True)
train_with_specs.drop('game_session', axis=1, inplace=True)
train_with_specs['time'] = pd.to_datetime(train_with_specs['timestamp'])

train_with_specs['weekday'] = train_with_specs['time'].dt.weekday

train_with_specs['hours'] = train_with_specs['time'].dt.hour
sns.distplot(train_with_specs['weekday'])
ax = sns.barplot(x='weekday', y='accuracy_group', data=train_with_specs)
print('Correlation on full dataset:', 

      round(train_with_specs['accuracy_group'].corr(train_with_specs['weekday']), 2))
train_with_specs.drop('weekday', axis=1, inplace=True)
sns.distplot(train_with_specs['hours'])
ax = sns.barplot(x='hours', y='accuracy_group', data=train_with_specs, color="salmon", saturation=.6)
print('Correlation on full dataset:', 

      round(train_with_specs['accuracy_group'].corr(train_with_specs['hours']), 2))
train_with_specs.drop('hours', axis=1, inplace=True)
train_with_specs.drop('event_data', axis=1, inplace=True)
train_with_specs.drop('installation_id_x', axis=1, inplace=True)

train_with_specs.drop('installation_id_y', axis=1, inplace=True)
train_with_specs = train_with_specs.loc[train_with_specs['event_count'] <= 200]
sns.distplot(train_with_specs['event_count'])
sns.relplot(x="event_count", y="accuracy_group", kind="line", data=train_with_specs);
print('Correlation on full dataset:', round(train_with_specs['accuracy_group'].corr(train_with_specs['event_count']), 2))
bins = pd.IntervalIndex.from_tuples([(0, 2500), (2500, 3500), (3500, 5000)])

train_with_specs['event_code'] = le.fit_transform(pd.cut(train_with_specs['event_code'], bins))
sns.distplot(train_with_specs['event_code'])
sns.relplot(x="event_count", y="accuracy_group", kind="line", data=train_with_specs);
print('Correlation on full dataset:', 

      round(train_with_specs['accuracy_group'].corr(train_with_specs['event_count']), 2))
train_with_specs.drop('game_time', axis=1, inplace=True)
train_with_specs.drop('timestamp', axis=1, inplace=True)
train_with_specs.drop('title_y', axis=1, inplace=True)
le.fit(['Game', 'Assessment', 'Activity', 'Clip'])

train_with_specs['type'] = le.transform(train_with_specs['type'])
le.fit(['NONE', 'TREETOPCITY', 'MAGMAPEAK', 'CRYSTALCAVES'])

train_with_specs['world'] = le.transform(train_with_specs['world'])
print('Correlation on full dataset:', round(train_with_specs['accuracy_group'].corr(train_with_specs['world']), 2))
train_with_specs.drop('time', axis=1, inplace=True)

train_with_specs.drop('num_correct', axis=1, inplace=True)

train_with_specs.drop('num_incorrect', axis=1, inplace=True)

train_with_specs.drop('accuracy', axis=1, inplace=True)
test_with_specs = test.join(specs.set_index('event_id'), on='event_id')

print('Num of unique values in "info" column:', test_with_specs['info'].nunique())

print('Num of unique values in "args" column:', test_with_specs['args'].nunique())
test_with_specs.drop('info', axis=1, inplace=True)

test_with_specs['args'] = le.fit_transform(test_with_specs['args'])

test_with_specs.head(2)
sns.distplot(test_with_specs['args'])
test_with_specs.drop('event_id', axis=1, inplace=True)
test_with_specs.drop('game_session', axis=1, inplace=True)
test_with_specs.drop('timestamp', axis=1, inplace=True)
test_with_specs.drop('event_data', axis=1, inplace=True)
sns.distplot(test_with_specs['event_code'])
bins = pd.IntervalIndex.from_tuples([(0, 2500), (2500, 3500), (3500, 8000)])

test_with_specs['event_code'] = le.fit_transform(pd.cut(test_with_specs['event_code'], bins))
sns.distplot(test_with_specs['event_code'])
test_with_specs.drop('game_time', axis=1, inplace=True)
test_with_specs['title'] = le.fit_transform(test_with_specs['title'])
le.fit(['Game', 'Assessment', 'Activity', 'Clip'])

test_with_specs['type'] = le.transform(test_with_specs['type'])
le.fit(['NONE', 'TREETOPCITY', 'MAGMAPEAK', 'CRYSTALCAVES'])

test_with_specs['world'] = le.transform(test_with_specs['world'])
test_with_specs.info()
train_with_specs.info()
X = train_with_specs.drop('accuracy_group', axis=1)

Y = train_with_specs['accuracy_group']
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)

clf = clf.fit(X, Y)
test_with_specs['session_predictions'] = clf.predict(test_with_specs.drop('installation_id', axis=1))
import math

submit = pd.DataFrame(np.ceil(test_with_specs.groupby(['installation_id'])['session_predictions'].mean()))
submit['installation_id'] = submit.index
for col in submit.columns: 

    print(col) 
cols = submit.columns.tolist()

cols = cols[-1:] + cols[:-1]

submit = submit[cols]
submit.rename(columns={'session_predictions': 'accuracy_group'}, inplace=True)
submit.rename(columns={'installation_id': 'id'}, inplace=True)
submit.reset_index(inplace=True)
submit.drop('id', axis=1, inplace=True)
submit
submit['accuracy_group'] = submit['accuracy_group'].astype(int)
submit
submit.to_csv('submission.csv', index=False)