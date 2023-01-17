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


import datetime
from time import time
from tqdm import tqdm_notebook as tqdm
from collections import Counter
from scipy import stats

from sklearn.model_selection import GroupKFold
from typing import Any
from numba import jit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from itertools import product
import copy
import time

import random
seed = 1234
random.seed(seed)
np.random.seed(seed)
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
train.head()
train.shape
train.columns
# Select all Assessment from installation_id
keep_id = train[train["type"] == "Assessment"]["installation_id"].drop_duplicates()

train = pd.merge(train, keep_id, on="installation_id", how="inner")
train.shape
keep_id.shape
train.type.value_counts()
fig = plt.figure(figsize=(12, 10))

ax1 = fig.add_subplot(211)
ax1 = sns.countplot(y="type", data=train, order= train.type.value_counts().index)
plt.title("number of events by type")

ax2 = fig.add_subplot(212)
ax2 = sns.countplot(y="world", data=train, order = train.world.value_counts().index)
plt.title("number of events by world")
plt.tight_layout(pad=0)
fig = plt.figure(figsize=(12,10))

title_count = train.title.value_counts().sort_values(ascending=True)
title_count.plot.barh()
plt.title("Event counts by title")
train['timestamp'] = pd.to_datetime(train['timestamp'])
train['date'] = train['timestamp'].dt.date
train['month'] = train['timestamp'].dt.month
train['hour'] = train['timestamp'].dt.hour
train['dayofweek'] = train['timestamp'].dt.dayofweek
fig = plt.figure(figsize=(12,10))
date = train.groupby('date')['date'].count()
date.plot()
plt.xticks(rotation=90)
plt.title("Event counts by date")
fig = plt.figure(figsize=(12,10))
day_of_week = train.groupby('dayofweek')['dayofweek'].count()
# convert num -> category
day_of_week.index = list(calendar.day_abbr)
day_of_week.plot.bar()
plt.title("Event counts by day of week")
plt.xticks(rotation=0)
fig = plt.figure(figsize=(12,10))
hour = train.groupby('hour')['hour'].count()
hour.plot.bar()
plt.title("Event counts by hour of day")
plt.xticks(rotation=0)
test.head()
test.shape
test.installation_id.nunique()
sample_submission.shape
set(list(train.installation_id.unique())).intersection(list(test.installation_id.unique()))
test['timestamp'] = pd.to_datetime(test['timestamp'])

print(f'The range of date in train is: {train.date.min()} to {train.date.max()}')
print(f'The range of date in test is: {test.timestamp.dt.date.min()} to {test.timestamp.dt.date.max()}')
train_labels.head()
train_labels.shape
pd.crosstab(train_labels.title, train_labels.accuracy_group)
plt.figure(figsize=(12,6))
sns.countplot(y="title", data=train_labels, order = train_labels.title.value_counts().index)
plt.title("Counts of titles")
df = train_labels.groupby(['accuracy_group', 'title'])['accuracy_group']
df.count()
se = train_labels.groupby(['title', 'accuracy_group'])['accuracy_group'].count().unstack('title')
se.plot.bar(stacked=True, rot=0, figsize=(12,10))
plt.title("Counts of accuracy group")
train[~train.installation_id.isin(train_labels.installation_id.unique())].installation_id.nunique()
train = train[train.installation_id.isin(train_labels.installation_id.unique())]
train.shape
print(f'No. of rows in train_labels: {train_labels.shape[0]}')
print(f'Number of unique game_sessions in train_labels: {train_labels.game_session.nunique()}')
train = train.drop(['date', 'month', 'hour', 'dayofweek'], axis=1)
def encode_title(train, test, train_labels):
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    
    all_title_event_code = list(set(train['title_event_code'].unique()).union(test['title_event_code'].unique()))
    
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    
    activities_map = dict(zip(list_of_user_activities, np,arange(len(list_of_user_activities))))
    
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    
    # replace title with its number from the dictionary
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    
    train_labels['title'] = train_labels['title'].map(activities_map)
    
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    return train, test, train_labels, all_title_event_code, list_of_user_activities, list_of_event_code, list_of_event_id, activities_labels, assess_titles
    
train, test, train_labels, all_title_event_code, list_of_user_activities, list_of_event_code, list_of_event_id, activities_labels, assess_titles = encode_title(train, test, train_labels)