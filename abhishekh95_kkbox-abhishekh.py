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

members = pd.read_csv('../input/members.csv')
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
df = train.merge(members,how='inner',on='msno')

df = df.merge(songs,how='inner',on='song_id')

df.drop_duplicates(subset =['source_system_tab', 'source_screen_name',

       'source_type', 'target', 'city', 'bd', 'gender', 'registered_via',

       'registration_init_time', 'expiration_date', 'song_length', 'genre_ids',

       'artist_name', 'composer', 'lyricist', 'language'], 

                     keep = False, inplace = True) 

df.reset_index(drop = True,inplace=True)

df
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
f,axarray = plt.subplots(1,1,figsize=(15,10))

plt.xlabel('Number of song plays')

plt.ylabel('Chance of repeat listens')

plt.plot(x_plays, y_repeat_chance)
songs.head()
train.columns
temp=df[['msno','song_id','source_type']]

temp_local=temp.where(temp['source_type']=='local-playlist')

local_percent=(temp_local.dropna(how='any').shape[0])/temp.shape[0]

temp_online=temp.where(temp['source_type']=='online-playlist')

online_percent=(temp_online.dropna(how='any').shape[0])/temp.shape[0]

temp_local_library=temp.where(temp['source_type']=='local-library')

local_library_percent = (temp_local_library.dropna(how='any').shape[0])/temp.shape[0]

temp_top_hits =temp.where(temp['source_type']=='top-hits-for-artist')

top_hits_percent = (temp_top_hits.dropna(how='any').shape[0])/temp.shape[0]

temp_album =temp.where(temp['source_type']=='album')

album_percent = (temp_album.dropna(how='any').shape[0])/temp.shape[0]

temp_nan =temp.where(temp['source_type']=='nan')

nan_percent = (temp_nan.dropna(how='any').shape[0])/temp.shape[0]

temp_song_based_playlist = temp.where(temp['source_type']=='song-based-playlist')

song_based_playlist_percent = (temp_song_based_playlist.dropna(how='any').shape[0])/temp.shape[0]

temp_radio = temp.where(temp['source_type']=='radio')

radio_percent = (temp_radio.dropna(how='any').shape[0])/temp.shape[0]

temp_song = temp.where(temp['source_type']=='song')

song_percent = (temp_song.dropna(how='any').shape[0])/temp.shape[0]

temp_listen_with = temp.where(temp['source_type']=='listen-with ')

listen_with_percent = (temp_listen_with.dropna(how='any').shape[0])/temp.shape[0]

temp_artist = temp.where(temp['source_type']=='artist')

artist_percent = (temp_artist.dropna(how='any').shape[0])/temp.shape[0]

temp_topic_article_playlist = temp.where(temp['source_type']=='topic-article-playlist')

article_playlist_percent = (temp_topic_article_playlist.dropna(how='any').shape[0])/temp.shape[0]

temp_my_daily_playlist = temp.where(temp['source_type']=='my-daily-playlist')

my_daily_playlist_percent = (temp_my_daily_playlist.dropna(how='any').shape[0])/temp.shape[0]



temp['source_type'].unique()
total_percentage=local_percent+online_percent+local_library_percent+top_hits_percent+album_percent+nan_percent+song_based_playlist_percent+radio_percent+song_percent+listen_with_percent+artist_percent+article_playlist_percent+my_daily_playlist_percent
source_type_percentage=[online_percent,local_percent,local_library_percent,top_hits_percent,album_percent,nan_percent,song_based_playlist_percent,radio_percent,song_percent,listen_with_percent,artist_percent,article_playlist_percent,my_daily_playlist_percent]
other = top_hits_percent+album_percent+nan_percent+song_based_playlist_percent+radio_percent+song_percent+listen_with_percent+artist_percent+article_playlist_percent+my_daily_playlist_percent
modified_names=["online_percent","local_percent","local_library_percent","other"]

modified_source_type = [online_percent,local_percent,local_library_percent,other]
modified_source_type
plt.subplot(1,2,1)

sns.barplot(x=temp['source_type'].unique(),y=source_type_percentage)

plt.xticks(rotation=90)

plt.subplot(1,2,2)

sns.barplot(x=modified_names,y=modified_source_type)

plt.xticks(rotation=90)

(members['gender'].isna().sum())/members.shape[0]
songs.columns
def count_vals(x):

    # count number of values (since we can have mutliple values separated by '|')

    if type(x) != str:

        return 1

    else:

        return 1 + x.count('|')
song_data['number_of_genres']=song_data['genre_ids'].apply(count_vals)

song_data['number_of_composers'] = song_data['composer'].apply(count_vals)

song_data['number_of_lyricists'] = song_data['lyricist'].apply(count_vals)

new_song_data = pd.DataFrame() 
new_song_data['song_id']=song_data['song_id'] 

new_song_data["Number_of_plays"]=song_data['plays']

new_song_data['Number_of_genre']=song_data['number_of_genres']

new_song_data['Number_of_composer']=song_data['number_of_composers']

new_song_data['Number_of_lyricist']=song_data['number_of_lyricists']

new_song_data['Language']=songs.language
new_song_data = new_song_data.sort_values(by = 'Number_of_plays', ascending=False)

    
#songs_info['song_id'].head()

#new_song_data.head()

new_new_song_data = pd.merge(new_song_data, songs_info, how='inner', on='song_id')

new_new_song_data.head()
songs.head
df.head()

sa=df[['msno','song_id','artist_name']]

sa1=df[['msno','song_id']]



sa = sa.groupby(['msno','artist_name']).count()

sa1 = sa1.groupby('msno').count()

sa1







sa.reset_index('artist_name',inplace=True)



sa



final = sa.merge(sa1, left_index=True, right_index=True, how='left')



final['normalized']=final['song_id_x']/final['song_id_y']



final.reset_index('msno',inplace=True)



final = final[['msno','artist_name','normalized']]

final=final.groupby("msno", group_keys=False).apply(lambda g: g.nlargest(20, "normalized"))
#final.set_index(['msno','artist_name'],inplace=True)
final



#final.pivot(index='msno', columns='artist_name', values='normalized')
piv=final.pivot(index='msno',columns='artist_name',values='normalized')
piv['Various Artists'].notnull().sum()