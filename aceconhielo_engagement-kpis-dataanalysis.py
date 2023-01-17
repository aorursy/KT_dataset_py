# Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

# Load Data without event_data

path_data = '../input/data-science-bowl-2019/'

columns = ['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code','game_time', 'title', 'type', 'world']



train = pd.read_csv(path_data + 'train.csv', usecols=columns) # no event_data

#  test = pd.read_csv(path_data + 'test.csv', usecols=columns) # no event_data

samplesubmission = pd.read_csv(path_data + 'sample_submission.csv')

train_labels = pd.read_csv(path_data + 'train_labels.csv')
samplesubmission.head()
samplesubmission.groupby('accuracy_group', as_index=False).count()
# 1. How many installation_id uniques are there?

iid_unique = list(set(train_labels.installation_id.values))

print('There are ', len(iid_unique), ' unique installations_id')

print('===================')

# 2. How many title uniques?

title_unique = list(set(train_labels.title.values))

print('There are ', len(title_unique), ' unique titles:')

for tu in title_unique:

    print('Title: ', tu)

    

# 3. Ranges for num_correct, num_incorrect, accuracy, accuracy_group

print('===================')

print(train_labels.describe())
# 4. Accuracy_group == (num_correct + num_incorrect) / num_correct?

train_labels['Accuracy_Group_Check'] = train_labels.apply(lambda x: x['num_correct']/(x['num_correct'] + x['num_incorrect']) if x['num_correct']>0 else 0, axis = 1)

train_labels[['Accuracy_Group_Check','accuracy', 'accuracy_group']].drop_duplicates()
# 5. Relation bwetween accuracy and accuracy_group

train_labels[['accuracy', 'accuracy_group']].drop_duplicates()

dataplot = train_labels[['accuracy', 'accuracy_group']].drop_duplicates()

sns.jointplot(x='accuracy', y='accuracy_group', data=dataplot)

plt.show()
train.info()
train.head()
train.isnull().sum()
print('installation_id check. In train but not in train_labels', len(list(set(train[train['title'].str.contains('Assessment')].installation_id.values) - set(train_labels.installation_id.values))))

print('installation_id check. In train_labels but not in train', len(list(set(train_labels.installation_id.values) - set(train[train['title'].str.contains('Assessment')].installation_id.values))))
print('# of event_id and game_session keys', train[['event_id','game_session','game_time','installation_id']].shape)

print('# of event_id and game_session keys with duplicates',train[train.duplicated(['event_id','game_session','game_time','installation_id'])].shape)
print('Conversion of duplicates: ', round(173517/11341042*100,2),'%')
trainduplicates = train[train.duplicated(['event_id','game_session','game_time'])].sort_values(['event_id','game_session','game_time','installation_id'], ascending=True).head()
train[(train['event_id']=='0086365d') & (train['game_session'] == 'e733c635dcb92407')]
print('Installation Id with duplicates: ')

print(set(trainduplicates.installation_id.values))
unique_titles = sorted(list(set(train.title.values)))

unique_types = sorted(list(set(train.type.values)))

unique_worlds = sorted(list(set(train.world.values)))



print('-- Unique titles: ', unique_titles)

print('\n-- Unique types: ', unique_types)

print('\n-- Unique worlds: ', unique_worlds)



print('--- NONE world analysis ---')

print('# Instances: ', train[train['world'] == 'NONE'].shape)

print('# Unique installation_id: ', len(set(train[train['world'] == 'NONE'].installation_id.values)))

print('# Unique title: ', set(train[train['world'] == 'NONE'].title.values))

sns.set(rc={'figure.figsize':(25,15)})

sns.barplot(x='event_id', y='title', data=train[['event_id','title']].groupby('title', as_index=False).count().sort_values('event_id',ascending=False))
# Range for assessments

train[train['title'].str.contains('Assessment')][['event_id','title']].groupby('title',as_index=False).count().sort_values('event_id',ascending=False)
# Range for games levels

train[train['title'].str.contains('Level')][['event_id','title']].groupby('title',as_index=False).count().sort_values('event_id',ascending=False)
train[['event_id','world']].groupby('world',as_index=False).count().sort_values('event_id',ascending=False)
train[['event_id','title','world']][train['title']=='Welcome to Lost Lagoon!'].groupby(['title','world'],as_index=False).count()
def wrangle_time(df, namedf):

      """

      :param df: dataframe where you are going to apply the time transformations

      :param namedf: name of the above dataframe

      :return: dataframe with time column transformed in a good formated, new column date, new column hour and new

      column weekday

      """

      print('Wrangling time in dataframe: ' + namedf + '...')

      df['timestamp'] = pd.to_datetime(df['timestamp'])

      df['date'] = df['timestamp'].dt.date

      df['month'] = df['timestamp'].dt.month

      df['hour'] = df['timestamp'].dt.hour

      df['weekday'] = df['timestamp'].dt.weekday_name

      print('Finished')

      return df





train = wrangle_time(train, 'TRAIN')

train_time = train[['event_id','timestamp','date','month','hour','weekday']]
min_date = min(train_time.date)

max_date = max(train_time.date)

print('Min Date: ', min_date)

print('Max Date: ', max_date)
plotdate = train_time[['event_id','date']].groupby('date', as_index=False).count()

plotmonth = train_time[['event_id','month']].groupby('month', as_index=False).count()

plothour = train_time[['event_id','hour']].groupby('hour', as_index=False).count()

plotweekday = train_time[['event_id','weekday']].groupby('weekday', as_index=False).count()
plotdate['dateformat'] = plotdate.apply(lambda x: x['date'].strftime("%b %d - %a"), axis=1)



plotmonth['monthL'] = plotmonth['month'].map({7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'})

plotmonth = plotmonth.sort_values('month', ascending=True)



plotweekday['weekdayn'] = plotweekday['weekday'].map({'Monday':0,'Tuesday':1, 'Wednesday':2,'Thursday':3, 'Friday':4,'Saturday':5, 'Sunday':6})

plotweekday = plotweekday.sort_values('weekdayn', ascending=True)
sns.set(rc={'figure.figsize':(25,7)})

axdate = sns.lineplot(x='dateformat', y='event_id', data=plotdate, sort=False)

for item in axdate.get_xticklabels():

    item.set_rotation(90)
# Set up the matplotlib figure

f, axes = plt.subplots(2, 2, figsize=(20, 7), sharex=False)

sns.despine(left=True)



sns.lineplot(x='date', y='event_id', data=plotdate, ax=axes[0, 0])

sns.lineplot(x='monthL', y='event_id', data=plotmonth, sort=False, ax=axes[0,1])

sns.lineplot(x='hour', y='event_id', data=plothour, ax=axes[1,0])

sns.lineplot(x='weekday', y='event_id', data=plotweekday, sort=False, ax=axes[1,1])
engagement_metrics = train[['installation_id','date']].drop_duplicates().groupby('date', as_index=False).count().sort_values('date',ascending=True)

engagement_metrics = engagement_metrics.rename(columns={"date": "date", "installation_id": "DAU"})
engagement_metrics['WAU'] = engagement_metrics['DAU'].rolling(min_periods=7, window=7).sum()

engagement_metrics['MAU'] = engagement_metrics['DAU'].rolling(min_periods=28, window=28).sum()



engagement_metrics['StickinessWAU'] = engagement_metrics['DAU']/engagement_metrics['WAU']

engagement_metrics['StickinessMAU'] = engagement_metrics['DAU']/engagement_metrics['MAU']





engagement_metrics['dateformat'] = engagement_metrics.apply(lambda x: x['date'].strftime("%b %d - %a"), axis=1)

engagement_metrics.describe()
sns.set(rc={'figure.figsize':(25,7)})

axdau = sns.lineplot(x='dateformat', y='DAU', data=engagement_metrics, sort=False)

for item in axdau.get_xticklabels():

    item.set_rotation(90)
# Set up the matplotlib figure

f, axes = plt.subplots(1, 2, figsize=(30, 10), sharex=False)

axwau = sns.lineplot(x='date', y='WAU', data=engagement_metrics, sort=False, ax=axes[0])

axmau = sns.lineplot(x='date', y='MAU', data=engagement_metrics, sort=False, ax=axes[1])



for item in axwau.get_xticklabels():

    item.set_rotation(90)

    

for item in axmau.get_xticklabels():

    item.set_rotation(90)   
