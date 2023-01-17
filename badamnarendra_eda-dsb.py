# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import calendar



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

train_labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
def timestamp_split(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['day_week'] = df['timestamp'].dt.dayofweek

    df['day_name']=df['timestamp'].dt.weekday_name

    return df

    

train = timestamp_split(train)
display(train_labels.head(),train_labels.shape)
train_labels['title'].value_counts(normalize=True)
print('Train data set has {} installation unique ids'.format(train['installation_id'].nunique()))

print('Train labels data set has {} installation unique ids'.format(train_labels['installation_id'].nunique()))





###Retaining only those installation id's required for training

req_id = train[train.type == "Assessment"][['installation_id']].drop_duplicates()

train= pd.merge(train, req_id, on="installation_id", how="inner")
print('After retaining only assesment data set we have {} rows'.format(train.shape[0]))



df2=pd.crosstab(train_labels['accuracy_group'],train_labels['title'])



df2=df2.reset_index()







fig,ax=plt.subplots(2,2)

fig.set_size_inches(15,15)

sns.countplot(train_labels['accuracy_group'],ax=ax[0,0]).set_title('Accuracy group distribution')





sns.countplot(train['type'],order=train.type.value_counts().index,ax=ax[0,1]).set_title('Type wise distribution ')



#plt.title('Activity Type by Dayname')

sns.set()

sns.countplot(train['day_name'],hue=train['type'],ax=ax[1,0]).set_title('Activity Type by Dayname')





plt.title('Accuracy group by Assesment')

sns.set()

df2.set_index('accuracy_group').T.plot(kind='bar', stacked=True,ax=ax[1,1],grid=False)

plt.xticks(rotation=45)









fig=plt.gcf()

fig.set_size_inches(8,8)

sns.countplot(y=train_labels['title'],order=train_labels.title.value_counts().index)

plt.title('Assessment frequency plot')
fig=plt.gcf()

fig.set_size_inches(20,20)

#plt.barh(train['title'],width=10)

sns.countplot(y=train['title'],order=train['title'].value_counts().sort_values(ascending=False).index)

plt.title('Title wise frequency')







fig = plt.gcf()

fig.set_size_inches(10,10)

ecd = train.date.value_counts()

ecd.plot()

plt.title("Even count by Date")

plt.xticks(rotation=60)

plt.show()
###Event comparision hour wise

fig=plt.figure(figsize=(12,10))



sns.countplot(train['hour'],color='skyblue')



plt.title('Hour-wise comparision of events')









fig=plt.gcf()

fig.set_size_inches(12,10)

week_freq=train.day_name.value_counts(sort=True)

# display(week_freq)



week_freq.plot.bar()

plt.title("Event frequency by dayname")

plt.xticks(rotation=30)

plt.show()
plt.figure(0)

plt.figure(figsize=(8,8))

title_game = train_labels.groupby(['title'])['game_session'].count().reset_index()

title_game['game_session'].plot.pie(explode=[0.05,0.05,0.05,0.05,0.05],autopct='%1.1f%%',labels=title_game['title'])

plt.title('Assesment wise session breakdown')





plt.figure(1)

plt.figure(figsize=(8,8))

atb=train['type'].value_counts().plot.pie(explode=[0.05,0.05,0.05,0.05],autopct='%1.1f%%')

atb.set_title('Activity Type % breakdown')





plt.figure(2)

plt.figure(figsize=(15,8))

accuracy_grp = train_labels.groupby(['accuracy_group'])['game_session'].count().reset_index()

agp=accuracy_grp['game_session'].plot.pie(explode=[0.05,0.05,0.05,0.05],autopct='%1.1f%%',labels=accuracy_grp['accuracy_group'])

agp.set_title('Accuracy group % breakdown')








