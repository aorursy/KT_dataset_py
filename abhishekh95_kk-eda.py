# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

songs = pd.read_csv('../input/songs.csv')

test = pd.read_csv('../input/test.csv')

songs_info = pd.read_csv('../input/song_extra_info.csv')

songs_info = pd.read_csv('../input/song_extra_info.csv')

members_old = pd.read_csv('../input/members.csv')
print('Song stats: ')

songs_in_train_and_test = np.intersect1d(train['song_id'].unique(), test['song_id'].unique())

print("Number of unique songs in our training: %d and test data: %d"% (train['song_id'].nunique(), test['song_id'].nunique()))

print("Songs in the test set do no appear in the train set ",test['song_id'].nunique() - songs_in_train_and_test.shape[0])

print("% of the test set size:",(test['song_id'].nunique() - songs_in_train_and_test.shape[0]) / test['song_id'].nunique())

print('User stats: ')

users_in_train_and_test = np.intersect1d(train['msno'].unique(), test['msno'].unique())

print("Number of unique users in our training: %d and test data: %d"% (train['msno'].nunique(), test['msno'].nunique()))

print("Users in the test set do no appear in the train set ",(test['msno'].nunique() - users_in_train_and_test.shape[0]))

print("% of the test set size:",(test['msno'].nunique() - users_in_train_and_test.shape[0]) / test['msno'].nunique())

train_merged = train.merge(songs[['song_id', 'artist_name', 'genre_ids',

                                       'language']], on='song_id')

test_merged = test.merge(songs[['song_id', 'artist_name', 'genre_ids',

                                     'language']], on='song_id')

print('Artists stats: ')

artists_in_train_and_test = np.intersect1d(train_merged['artist_name'].unique(),

                                           test_merged['artist_name'].unique())

print("Number of unique artists in our training: %d and test data: %d"% (train_merged['artist_name'].nunique(), test_merged['artist_name'].nunique()))

print("Artists in the test set do no appear in the train set ",(test_merged['artist_name'].nunique() - artists_in_train_and_test.shape[0]))

print("% of the test set size:",(test_merged['artist_name'].nunique()

       - artists_in_train_and_test.shape[0]) / test_merged['artist_name'].nunique())
print('Language stats: ')

langs_in_train_and_test = np.intersect1d(train_merged['language'].unique(),

                                          test_merged['language'].unique())

print("Number of unique languages in our training: %d and test data: %d"% (train_merged['language'].nunique(), test_merged['language'].nunique()))

print("Languages in the test set do no appear in the train set ",(test_merged['language'].nunique() - langs_in_train_and_test.shape[0]))

print("% of the test set size:",(test_merged['language'].nunique()

       - langs_in_train_and_test.shape[0]) / test_merged['language'].nunique())
print('Genre stats: ')

genres_in_train_and_test = np.intersect1d(train_merged['genre_ids'].apply(str).unique(),

                                          test_merged['genre_ids'].apply(str).unique())

print("Number of unique Genres in our training: %d and test data: %d"%(train_merged['genre_ids'].nunique(), test_merged['genre_ids'].nunique()))

print("Genres in the test set do no appear in the train set ",(test_merged['genre_ids'].nunique() - genres_in_train_and_test.shape[0]))

print("% of the test set size:",(test_merged['genre_ids'].nunique()

       - genres_in_train_and_test.shape[0]) / test_merged['genre_ids'].nunique())
sample_length = int(.50*len(members_old))

members = members_old.sample(n = sample_length, random_state = 1)

df = train.merge(members,how='inner',on='msno')

df = df.merge(songs,how='inner',on='song_id')

df.drop_duplicates(subset =['source_system_tab', 'source_screen_name',

       'source_type', 'target', 'city', 'bd', 'gender', 'registered_via',

       'registration_init_time', 'expiration_date', 'song_length', 'genre_ids',

       'artist_name', 'composer', 'lyricist', 'language'], 

                     keep = False, inplace = True) 

df.reset_index(drop = True,inplace=True)
listen_log = df[['msno','song_id','target']].merge(songs,on='song_id')

listen_log_groupby = listen_log[['song_id', 'target']].groupby(['song_id']).agg(['mean','count'])

listen_log_groupby.reset_index(inplace=True)

listen_log_groupby.columns = list(map(''.join, listen_log_groupby.columns.values))

listen_log_groupby.columns=['song_id','repeat_play_chance', 'plays']

song_data = listen_log_groupby.merge(songs, on='song_id')

song_data['repeat_events'] = song_data['repeat_play_chance'] * song_data['plays']
x_plays = []

y_repeat_chance = []



for i in range(1,song_data['plays'].max()+1):

    plays_i = song_data[song_data['plays']==i]

    count = plays_i['plays'].sum()

    if count > 0:

        x_plays.append(i)

        y_repeat_chance.append(plays_i['repeat_events'].sum() / count)
plt.figure(figsize=(45,12))

plt.title('Number of Null Values in each feature',size=50)

null_columns=df.columns[df.isnull().any()]

(df[null_columns].isnull().sum()/df.shape[0]).sort_values(ascending=False).plot(kind='bar')

plt.xticks(size = 30)

plt.yticks(size = 30)
f,axarray = plt.subplots(1,1,figsize=(10,7))

plt.title('Relation between Popularity and Chance of replays',size=20)

plt.xlabel('Number of song plays')

plt.ylabel('Chance of repeat listens')

plt.plot(x_plays, y_repeat_chance)
temp= df[['msno','song_id','source_type']]

fig = plt.figure(figsize=(45,5)) 

plt.title('Source type distribution',size=20)

plt.subplot(1,2,1)

(temp['source_type'].value_counts()[0:7]/temp.shape[0]).plot(kind='bar',title='Source type distribution')

languages = song_data['language'].unique()

print(languages.shape[0])



language_count = []

language_plays = []

language_repeat_chance = []



for l in languages:

    if not np.isnan(l):

        songs_with_language = song_data[song_data['language']==l]

        count = songs_with_language['plays'].sum()

        language_repeat_chance.append(songs_with_language['repeat_events'].sum() / count)

        language_count.append(songs_with_language.shape[0])

        language_plays.append(count)

    else:

        songs_with_language = song_data[pd.isnull(song_data['language'])]

        count = songs_with_language['plays'].sum()

        language_repeat_chance.append(songs_with_language['repeat_events'].sum() / count)

        language_count.append(songs_with_language.shape[0])

        language_plays.append(count)

        

languages[10] = -100  # we'll replace the nan value with something different



fig = plt.figure(figsize=(20, 12)) 

ax1 = plt.subplot(3,1,1)

plt.title('Relationship between Language and Number of Songs',size=20)

sns.barplot(x=languages,y=np.log10(language_count))

ax1.set_ylabel('log10(# of songs)')

ax2 = plt.subplot(3,1,2)

plt.title('Relationship between Language and Popularity',size=20)

sns.barplot(x=languages,y=np.log10(language_plays))

ax2.set_ylabel('log10(# of plays)')

ax3 = plt.subplot(3,1,3)

plt.title('Relationship between Language and Chance of replay',size=20)

sns.barplot(x=languages,y=language_repeat_chance)

ax3.set_ylabel('Chance of repeated listen')

ax3.set_xlabel('Song language')
artist_replgroupby = song_data[['artist_name', 'plays', 'repeat_events']].groupby(['artist_name'])

artist_replgroupby = artist_replgroupby['plays', 'repeat_events'].agg(['sum', 'count'])

artist_replgroupby.reset_index(inplace=True)

artist_replgroupby.columns = list(map(''.join, artist_replgroupby.columns.values))

artist_replgroupby.drop(['repeat_eventscount'], axis=1, inplace=True)

artist_replgroupby.columns = ['artist', 'plays', 'tracks', 'repeat_events']

artist_replgroupby['repeat_play_chance'] = artist_replgroupby['repeat_events'] / artist_replgroupby['plays']
artist_groupby = song_data[['artist_name', 'plays']].groupby(['artist_name'])

artist_plays = artist_groupby['plays'].agg(['sum'])

artist_plays.reset_index(inplace=True)



min_plays = artist_plays['sum'].min()

max_plays = artist_plays['sum'].max()

print(min_plays, max_plays)
plt.figure(figsize=(15,8))

play_bins = np.logspace(np.log10(min_plays),np.log10(max_plays+1),100)

# track_bins = np.linspace(1,max_tracks+1,100)

sns.distplot(artist_plays['sum'], bins=play_bins, kde=False,

             hist_kws={"alpha": 1})

plt.xlabel('# of plays')

plt.ylabel('# of artists')

plt.yscale('log')

plt.xscale('log')
plt.figure(figsize=(15,8))

chance_bins = np.linspace(0,1,100)

sns.distplot(artist_replgroupby['repeat_play_chance'], bins=chance_bins, kde=False,

             hist_kws={"alpha": 1})

plt.xlabel('Chance of repeated listens')

plt.ylabel('# of artists')

plt.yscale('log')

# plt.xscale('log')
artist_groupby = song_data[['artist_name', 'plays']].groupby(['artist_name'])

artist_plays = artist_groupby['plays'].agg(['sum'])

artist_plays.reset_index(inplace=True)



min_plays = artist_plays['sum'].min()

max_plays = artist_plays['sum'].max()

print(min_plays, max_plays)
plt.figure(figsize=(15,8))

df.genre_ids.value_counts()[:40].plot(kind='bar')

plt.xlabel('Genre_Id')

plt.ylabel('Count')

plt.xticks(rotation=90)
corr = df.corr()

sns.heatmap(corr,  xticklabels=corr.columns.values, yticklabels=corr.columns.values)

plt.title('feature correlations')
from wordcloud import WordCloud



def displaywc(txt,title):

    txt=""

    for i in g:

        txt+=str(i)

    wordcloud = WordCloud(background_color='white').generate(txt)

    plt.figure(figsize=(19,7))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.title(title)

    plt.show()
song_repeats=df.groupby('song_id',as_index=False).msno.count()

song_repeats.columns=['song_id','count']

song_repeats=pd.DataFrame(song_repeats).merge(songs,left_on='song_id',right_on='song_id')

g=song_repeats.sort_values(by='count',ascending=False)[:200].artist_name.tolist()
song_repeats.head()


#TODO Investigate how to display chinese

txt=""

for i in g:

    txt+=str(i)

displaywc(txt,'Most common artists that people listen to ')
#train.merge(songs_info,on='song_id').groupby('song_id')['name'].count().plot(kind='bar')
song_name=train.merge(songs_info,on='song_id')

song_repeats=song_name.groupby('name',as_index=False).song_id.count()

song_repeats.columns=['name','name_count']

g=song_repeats.sort_values(by='name_count',ascending=False)[:200].name.tolist()

# song_repeats.sort_values(by='count',ascending=False)[:200].artist_name.tolist()
#TODO Investigate how to display chinese

txt=""

for i in g:

    txt+=str(i)

displaywc(txt,'Most common song names that people listen to ')