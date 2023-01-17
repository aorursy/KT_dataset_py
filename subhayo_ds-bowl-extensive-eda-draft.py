import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

%%time

train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
np.random.seed(123)
train.sample(10)
train.shape
test.sample(10)
test.shape
assessed_only = train[train.type == 'Assessment'].drop_duplicates(subset='installation_id')[['installation_id']]

train = train[train.installation_id.isin(assessed_only['installation_id'])]

train.shape
len(set(train.installation_id.unique()) & (set(test.installation_id.unique())))
len(set(train.game_session.unique()) & (set(test.game_session.unique())))
len(set(train.event_id.unique()) & (set(test.event_id.unique())))
pd.options.display.max_colwidth = 150

specs.sample(10)
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype.name



        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:

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



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
reduce_mem_usage(train)
reduce_mem_usage(test)
reduce_mem_usage(specs)
reduce_mem_usage(train_labels)
plt.figure(figsize=(12,6))





sns.countplot(x='event_code',data=train, palette = 'Blues_d',

              order = train['event_code'].value_counts().index).set_title('Count by Event Code - Train')

plt.xticks(rotation=90,fontsize=8)

plt.show()
plt.figure(figsize=(12,6))





sns.countplot(x='event_code',data=test, palette = 'Blues_d',

              order = test['event_code'].value_counts().index).set_title('Count by Event Code - Test')

plt.xticks(rotation=90,fontsize=8)

plt.show()
train.groupby('event_code')[['event_count']].agg('sum').sort_values(by = 'event_count',ascending=False).head(10)
test.groupby('event_code')[['event_count']].agg('sum').sort_values(by = 'event_count',ascending=False).head(10)
train_labels.sample(10)
train = train[train.installation_id.isin(train_labels.installation_id.unique())]
train.shape
train.isnull().sum()
test.isnull().sum()
specs.isnull().sum()
train_labels.isnull().sum()
sns.set(font_scale=1.5,palette = 'Blues_d')

sns.set_style('whitegrid')

plt.figure(figsize=(12,6))





sns.countplot(x='type',data=train,

              order = train['type'].value_counts().index).set_title('Count by game type - Train')
sns.set(font_scale=1.5,palette = 'Blues_d')

sns.set_style('whitegrid')

plt.figure(figsize=(12,6))





sns.countplot(x='type',data=test,

              order = test['type'].value_counts().index).set_title('Count by game type - Test')
plt.figure(figsize=(12,7))



sns.countplot(x='world',data=train,

             order = train['world'].value_counts().index).set_title('Count by World - Train')
plt.figure(figsize=(12,7))



sns.countplot(x='world',data=test,

             order = test['world'].value_counts().index).set_title('Count by World - Test')
plt.figure(figsize=(12,12))



sns.countplot(y='title',data=train,palette = 'Blues_d',

             order = train['title'].value_counts().index).set_title('Count by Title - Train')
plt.figure(figsize=(12,12))



sns.countplot(y='title',data=test,palette = 'Blues_d',

             order = test['title'].value_counts().index).set_title('Count by Title - Test')
plt.figure(figsize=(12,6))



sns.countplot(y='title',data=train_labels,palette = 'Blues_d',

             order = train_labels['title'].value_counts().index).set_title('Count by Assessment - Train labels')
plt.figure(figsize=(12,6))



sns.countplot(y='accuracy_group',data=train_labels,palette = 'Blues_d',

             order = train_labels['accuracy_group'].value_counts().index).set_title('Count by Accuracy Group - Train labels')
train['timestamp'] = pd.to_datetime(train['timestamp'])

test['timestamp'] = pd.to_datetime(test['timestamp'])
plt.figure(figsize=(12,6))



sns.countplot(x=train['timestamp'].dt.dayofweek,data=train,palette = 'Blues_d').set_title('Count by Day of the Week - Train')
plt.figure(figsize=(12,6))



sns.countplot(x=test['timestamp'].dt.dayofweek,data=test,palette = 'Blues_d').set_title('Count by Day of the Week - Test')
plt.figure(figsize=(12,6))



sns.countplot(x=train['timestamp'].dt.hour,data=train,palette = 'Blues_d').set_title('Count by Hour of the Day - Train')
plt.figure(figsize=(12,6))



sns.countplot(x=test['timestamp'].dt.hour,data=test,palette = 'Blues_d').set_title('Count by Hour of the Day - Test')
train=train.sort_values('timestamp')

test=test.sort_values('timestamp')
plt.figure(figsize=(15,8))



sns.countplot(x=train['timestamp'].dt.date,data=train,palette = 'Blues_d').set_title('Count by Date - Train')

plt.xticks(rotation=90,fontsize=8)

plt.show()
plt.figure(figsize=(15,8))



sns.countplot(x=test['timestamp'].dt.date,data=test,palette = 'Blues_d').set_title('Count by Date - Test')

plt.xticks(rotation=90,fontsize=8)

plt.show()
train.shape
train_labels.shape
train_labels
train_new = pd.merge(train, train_labels.filter(['game_session','installation_id','accuracy_group'],axis=1), on=['installation_id','game_session'], how='left')
train = train_new
train.nunique()
grouped_events = train.groupby(['event_code'])['event_code'].count().rename('count').reset_index().sort_values('count', ascending=False)

grouped_events['perc'] = grouped_events['count'] / grouped_events['count'].sum()

grouped_events
grouped_events_test = test.groupby(['event_code'])['event_code'].count().rename('count').reset_index().sort_values('count', ascending=False)

grouped_events_test['perc'] = grouped_events_test['count'] / grouped_events_test['count'].sum()

grouped_events_test
set(grouped_events.head(6).event_code).intersection(grouped_events_test.head(6).event_code)
train['event_code'] = train['event_code'].apply(str)

test['event_code'] = test['event_code'].apply(str)
main_events = grouped_events.head(6).event_code

train[~train.event_code.isin(main_events)]
grouped_titles = train.groupby(['title'])['title'].count().rename('count').reset_index().sort_values('count', ascending=False)

grouped_titles['perc'] = grouped_titles['count'] / grouped_titles['count'].sum()

grouped_titles
grouped_titles_test = test.groupby(['title'])['title'].count().rename('count').reset_index().sort_values('count', ascending=False)

grouped_titles_test['perc'] = grouped_titles_test['count'] / grouped_titles_test['count'].sum()

grouped_titles_test
set(grouped_titles.head(6).title) & set(grouped_titles_test.head(6).title)
main_titles = grouped_titles.head(6).title

train[~train.title.isin(main_titles)]
def prepare_data(df):

    

    # Adding all the time columns

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['year'] = df['timestamp'].dt.year

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    

    # drop any unnecessary columns

    df = df.drop(['timestamp','event_data','game_session','event_id'], axis = 1)

    

    # merge all smaller event codes / titles together

    df.loc[(~df.event_code.isin(main_events),'event_code')]='0000'

    df.loc[(~df.title.isin(main_titles),'title')]='Other'

    

    # convert into dummy variables

    dummies = pd.get_dummies(df[['type','title','world','event_code']])

    

    # drop unnecessary columns

    df = df.drop(['type','title', 'world','event_code'], axis = 1)

    df = pd.concat([df, dummies], axis=1)

    

    return df
train_prep = prepare_data(train)

test_prep = prepare_data(test)
%whos DataFrame