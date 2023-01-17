import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import gc, sys, math, random

import datetime

import re

import json

from pandas.io.json import json_normalize

from tqdm._tqdm_notebook import tqdm_notebook as tqdm



import warnings

warnings.filterwarnings('ignore')



sns.set_style('darkgrid')



pd.options.display.float_format = '{:,.3f}'.format



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
%%time



train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv', converters={'args': json.loads})



train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
train.info()
train.head()
specs.info()
specs.head()
train_labels.info()
train_labels.head()
train['event_id'].nunique(), test['event_id'].nunique()
train['game_session'].nunique(), test['game_session'].nunique()
common_sesseion = set(train['game_session'].unique()) & set(test['game_session'].unique())

list(common_sesseion)
print('train: ', train['timestamp'].min(), 'to', train['timestamp'].max()) 

print('test:  ', test['timestamp'].min(), 'to', test['timestamp'].max())
_='''

START_DATE = '2019-07-23T00:00:00+0000'

startdate = pd.to_datetime(START_DATE)

train_day = train['timestamp'].apply(lambda x: (pd.to_datetime(x) - startdate).total_seconds() // (24 * 60 * 60))

train_day.hist()

'''
max_events_game_session = train[train['event_count'] == train['event_count'].max()]['game_session'].values[0]

subset = train[train['game_session'] == max_events_game_session][['game_session','event_count','timestamp','game_time']]

subset['timestamp'] = pd.to_datetime(subset['timestamp'])



min_ts = subset['timestamp'].min()

subset['timestamp_diff_sec'] = subset['timestamp'].apply(lambda x: int((x - min_ts).total_seconds() * 1000))

subset
train['installation_id'].nunique(), test['installation_id'].nunique()
common_installation = set(train['installation_id'].unique()) & set(test['installation_id'].unique())

list(common_installation)
game_session_count = train.groupby('installation_id')['game_session'].nunique()

game_session_count.describe()
train_corrected = train['event_data'].str.contains('"correct":true')

test_corrected = test['event_data'].str.contains('"correct":true')



fig, ax = plt.subplots(1, 2, figsize=(12,3))



train_subset = train[train_corrected]

train_subset['event_code'].value_counts().plot.bar(ax=ax[0])

print('train:', train_subset['type'].unique(), train_subset['event_code'].unique())



test_subset = test[test_corrected]

test_subset['event_code'].value_counts().plot.bar(ax=ax[1], color='limegreen')

print('test:', test_subset['type'].unique(), test_subset['event_code'].unique())
fig, ax = plt.subplots(1, 2, figsize=(12,3))



train_corrected.value_counts().plot.bar(ax=ax[0])

test_corrected.value_counts().plot.bar(ax=ax[1], color='limegreen')
train['event_count'].hist(bins=100,figsize=(12,3))

train['event_count'].describe().to_frame()
test['event_count'].hist(bins=100,figsize=(12,3), color='limegreen')

test['event_count'].describe().to_frame()
train['event_code'].nunique(), test['event_code'].nunique() 
fig, ax = plt.subplots(2, 1, figsize=(12,7))



train['event_code'].value_counts().plot.bar(ax=ax[0])

test['event_code'].value_counts().plot.bar(ax=ax[1], color='limegreen')
train['title'].nunique(), test['title'].nunique()
np.sort(train['title'].unique())
train['title'].value_counts().plot.bar(figsize=(12,3))
test['title'].value_counts().plot.bar(figsize=(12,3), color='limegreen')
train_title = train['title'].unique()

test_title = train['title'].unique()

[n for n in train_title if n not in test_title] + [n for n in test_title if n not in train_title]
# type & world is unique in same title

print('[title + type] unique count:', (train['title'] + '-' +  train['type']).nunique())

print('[title + world] unique count:', (train['title'] + '-' +  train['world']).nunique())
cols = ['world','type','title']

train[cols].drop_duplicates().sort_values(by=cols).reset_index(drop=True)
train['type'].unique()
fig, ax = plt.subplots(1, 2, figsize=(12,3))



train['type'].value_counts().plot.bar(ax=ax[0])

test['type'].value_counts().plot.bar(ax=ax[1], color='limegreen')
for t in train['type'].unique():

    print(f'events of {t}:\n', np.sort(train[train['type'] == t]['event_code'].unique()))
train[(train['event_code'] == 4100) | (train['event_code'] == 4110)]['type'].value_counts().plot.bar()
train['world'].unique()
fig, ax = plt.subplots(1, 2, figsize=(12,3))



train['world'].value_counts().plot.bar(ax=ax[0])

test['world'].value_counts().plot.bar(ax=ax[1], color='limegreen')
print('nunique event_id:', train['event_id'].nunique())

print('nunique event_code:', train['event_code'].nunique())

print('nunique event_id&event_code:', (train['event_id'] + '-' + train['event_code'].astype(str)).nunique())
train.groupby('event_code')['event_id'].unique().to_frame()
# start event

train[train['event_count'] == 1]['event_code'].value_counts()
subset = train[(train['event_code'] == 4100) | (train['event_code'] == 4110)]

subset
specs['event_id'].nunique(), len(specs)
specs['info'][0]
json_normalize(specs['args'][0])
print('max(args array size):', max(specs['args'].apply(len)))
max_colwidth = pd.get_option('display.max_colwidth')

max_rows = pd.get_option("display.max_rows")
_='''

pd.set_option('display.max_colwidth', 1000)

pd.set_option("display.max_rows", 500)



subset = train[['event_code','event_id']]

subset = subset[~subset.duplicated()]

event_info = pd.merge(subset, specs[['event_id','info']]).sort_values(by=['event_code','info'])

event_info.to_csv("event_info.csv", index=False)

display(event_info)



pd.set_option('display.max_colwidth', max_colwidth)

pd.set_option("display.max_rows", max_rows)

'''
train_labels.head()
max(train_labels['num_correct'] + train_labels['num_incorrect'])
print('[train] sessions:', train['game_session'].nunique(), ', installations:', train['installation_id'].nunique(), ', len:', len(train))

print('[test] sessions:', test['game_session'].nunique(), ', installations:', test['installation_id'].nunique(), ', len:', len(test))

print('[labels] sessions:', train_labels['game_session'].nunique(), ', installations:', train_labels['installation_id'].nunique(), ', len:', len(train_labels))
s = '77b8ee947eb84b4e'

subset = train[train['game_session'] == s]

subset['event_code'].value_counts().to_frame().T
s = '9501794defd84e4d'

subset = train[train['game_session'] == s]

subset['event_code'].value_counts().to_frame().T
'''

Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, 

which uses event_code 4110. If the attempt was correct, it contains "correct":true.

'''

# except for

train[(train['title'] == 'Bird Measurer (Assessment)') & ((train['event_code'] == '4100') | (train['event_code'] == '4110'))]
%%time



train['event_data.correct'] = ''

train.loc[train['event_data'].str.contains('"correct":true'), 'event_data.correct'] = 'true'

train.loc[train['event_data'].str.contains('"correct":false'), 'event_data.correct'] = 'false'



train['event_code_correct'] = train['event_code'].astype(str) + train['event_data.correct'].apply(lambda x: x if x == '' else f'({x})')

train_events = train.groupby('game_session').apply(lambda x: x['event_code_correct'].value_counts().to_frame().T.reset_index(drop=True))

train_events = train_events.reset_index().drop('level_1', axis=1)

train_events.head()
train_events['_event_correct'] = train_events[['2010','2030','4100(true)','4110(true)']].sum(axis=1) # ,3021,3121

train_events['_event_incorrect'] = train_events[['4100(false)','4110(false)']].sum(axis=1) #'3020','3120',

train_events.fillna(0, inplace=True)



session_events = pd.merge(train_labels, train_events)



cols = ['num_correct','num_incorrect']

session_events[cols + list(train_events.columns[1:])].corr()[cols].sort_index()
session_events['_calc_accuracy_group'] = 0

session_events.loc[(session_events['_event_incorrect'] == 0) & (session_events['_event_correct'] > 0),'_calc_accuracy_group'] = 3

session_events.loc[(session_events['_event_incorrect'] == 1) & (session_events['_event_correct'] > 0),'_calc_accuracy_group'] = 2

session_events.loc[(session_events['_event_incorrect'] >= 2) & (session_events['_event_correct'] > 0),'_calc_accuracy_group'] = 1



session_events['_calc_accuracy_group'].value_counts().plot.bar()
session_events[['accuracy_group','_event_correct','_event_incorrect','_calc_accuracy_group']]
ins_last_session = test.drop_duplicates(subset=['installation_id'], keep='last')

ins_last_session.head()
# last session contains only start-event

ins_last_session['event_count'].unique(), ins_last_session['event_code'].unique()
ins_last_session = ins_last_session[['installation_id','game_session']]

ins_last_session['last_session'] = 1



test = pd.merge(test, ins_last_session, how='left')

test['last_session'].fillna(0, inplace=True)



test['last_session'].value_counts()
print('last session events(mean): ', np.mean(test[test['last_session'] == 1].groupby('game_session')['event_id'].count()))