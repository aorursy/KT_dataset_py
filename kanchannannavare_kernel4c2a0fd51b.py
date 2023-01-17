# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import calendar



from sklearn.model_selection import  GroupKFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

train_labels = train_labels.drop(columns=['installation_id', 'title'])

train = pd.merge(train, train_labels, on='game_session')

# train['accuracy_group'] = train_labels['game_session'].apply(lambda x: x)

print(train.head())
print(train.shape)
keep_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()

train = pd.merge(train, keep_id, on="installation_id", how="inner")
print(train.shape)
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
plt.rcParams.update({'font.size': 12})



fig = plt.figure(figsize=(12,10))

se = train.title.value_counts().sort_values(ascending=True)

se.plot.barh()

plt.title("Event counts by title")

plt.xticks(rotation=0)

plt.show()
def get_time(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    return df

    

train = get_time(train)
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

se.plot.bar()

plt.title("Event counts by hour of day")

plt.xticks(rotation=0)

plt.show()
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')



print(test.head())



test.installation_id.nunique()
test['timestamp'] = pd.to_datetime(test['timestamp'])

print(f'The date range in train is: {train.timestamp.dt.date.min()} to {train.timestamp.dt.date.max()}')

print(f'The date range in test is: {test.timestamp.dt.date.min()} to {test.timestamp.dt.date.max()}')
plt.rcParams.update({'font.size': 22})



plt.figure(figsize=(12,6))

sns.countplot(y="title", data=train, color="blue", order = train.title.value_counts().index)

plt.title("Counts of titles")

plt.show()
plt.rcParams.update({'font.size': 16})



se = train.groupby(['title', 'accuracy_group'])['accuracy_group'].count().unstack('title')

se.plot.bar(stacked=True, rot=0, figsize=(12,10))

plt.title("Counts of accuracy group")

plt.show()
train[train.installation_id == "0006a69f"]
train[(train.event_code == 4100) & (train.installation_id == "0006a69f") & (train.title == "Bird Measurer (Assessment)")]
train[(train.installation_id == "0006a69f") & ((train.type == "Assessment") & (train.title == 'Bird Measurer (Assessment)') & (train.event_code == 4110) |

                                               (train.type == "Assessment") & (train.title != 'Bird Measurer (Assessment)') & (train.event_code == 4100))]
# train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))



# print(train.head())



# test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))



# all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    

# print(all_title_event_code)
true_attempts = train['event_data'].str.contains('true')



print(true_attempts)
print(type(train))
user_data = train[train['installation_id'].apply(lambda x: x == '0006a69f')].sort_values('timestamp', ascending=True)

# df.loc[df['column_name'] == some_value]

print(user_data)
print(user_data.shape[0])
 

# create data

x=range(1,user_data.shape[0]+1)

y=user_data['event_count']

 

# Change the color and its transparency

plt.fill_between( x, y, color="skyblue", alpha=0.4)

plt.show()

 

# Same, but add a stronger line on top (edge)

plt.fill_between( x, y, color="skyblue", alpha=0.2)

plt.plot(x, y, color="Slateblue", alpha=0.6)

print(user_data['event_id'].unique())
# df = sns.load_dataset(user_data)

 

# --- Use the 'palette' argument of seaborn

sns.lmplot( x="timestamp", y="event_count", data=user_data, fit_reg=False, hue='event_id', legend=True, palette="Set1")

# plt.legend(loc='lower right')

sns.lmplot( x="timestamp", y="event_count", data=user_data, fit_reg=False, hue='game_session', legend=True, palette="Set1")



sns.lmplot( x="timestamp", y="accuracy_group", data=user_data, fit_reg=False, hue='type', legend=True, palette="Set1")



print(type(train))
print(train_labels.head())
# new_train_labels = train_labels.drop(columns=['installation_id', 'title'])
# print(new_train_labels.head())
# train.join(new_train_labels.set_index('game_session'), on="game_session")

# pd.concat(train, train_labels)

print(train.head())
# new_train = pd.merge(train, train_labels, on='game_session')
sns.lmplot( x="timestamp", y="event_count", data=train, fit_reg=False, hue='accuracy_group', legend=True, palette="Set1")



# print(train_labels)
# get data

# train



###### variables

# event_count 

# game_session_count

# user_data['game_session_count'] = user_data[user_data['timestamp'].apply(lambda x: x)]

# list(map(lambda x, y: str(x) + '_' + str(y), train['timestamp'], train['event_code']))

# total_game_session

# total_accuracy

# avg_accuracy

# max_accuracy

# min_accuracy 

# total_event_count

# avg_event_count

# total_event_code

# max_event_code

# min_event_code

# total_num_correct

# avg_num_correct

# total_num_incorrect

# avg_num_incorrect

# total_game_time

# avg_game_time
params = {'n_estimators':2000,

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': 'rmse',

            'subsample': 0.75,

            'subsample_freq': 1,

            'learning_rate': 0.04,

            'feature_fraction': 0.9,

         'max_depth': 15,

            'lambda_l1': 1,  

            'lambda_l2': 1,

            'verbose': 100,

            'early_stopping_rounds': 100, 'eval_metric': 'cappa'

            }
# train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')





# y = train_labels['accuracy_group']
n_fold = 5

folds = GroupKFold(n_splits=n_fold)
cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']
# mt = MainTransformer()

# ft = FeatureTransformer()

# transformers = {'ft': ft}

# regressor_model1 = RegressorModel(model_wrapper=LGBWrapper_regr())

# regressor_model1.fit(X=train, y=y, folds=folds, params=params, eval_metric='cappa')
from sklearn.linear_model import LinearRegression



# Create linear regression object.

mlr= LinearRegression()



# Fit linear regression.

mlr.fit(train[['event_count']], train['accuracy_group'])



print(mlr.intercept_, mlr.coef_)

print(test.head() )
new_test = test.drop(columns=['event_id', 'game_session', 'timestamp', 'event_data', 'installation_id', 'event_code', 'game_time', 'title', 'type', 'world'])
new_test
pr = mlr.predict(new_test)



print(pr)

    
sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

print(sample_submission)

# sample_submission = []

# sample_submission['installation_id'] = test['installation_id']

# print(sample_submission)


d = {'installation_id' : test['installation_id'], 'accuracy_group': pr.astype(int)}

submission = pd.DataFrame(data=d)



print(submission)
# sample_submission['accuracy_group'] = pr.astype(int)



submission.to_csv('submission.csv', index=False)