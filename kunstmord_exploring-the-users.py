import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

songs = pd.read_csv('../input/songs.csv')

test = pd.read_csv('../input/test.csv')
train = train.merge(songs, on='song_id')

listen_log_groupby = train[['msno', 'target']].groupby(['msno']).agg(['count', 'sum'])

listen_log_groupby.reset_index(inplace=True)

listen_log_groupby.columns = list(map(''.join, listen_log_groupby.columns.values))



listen_log_groupby.columns = ['msno', 'plays', 'repeat_events']  #rename columns

train = listen_log_groupby.merge(train, on='msno') # merge song data with computed values

train['repeat_play_chance'] = train['repeat_events'] / train['plays']
print(train['plays'].max())
plt.figure(figsize=(15,8))

play_bins = np.linspace(0,train['plays'].max()+1,100)



sns.distplot(train['plays'], bins=play_bins, kde=False,

             hist_kws={"alpha": 1})

plt.xlabel('# of plays')

plt.ylabel('# of users')

# plt.yscale('log')

# plt.xscale('log')
plt.figure(figsize=(15,8))

rplay_bins = np.linspace(0,1.001,100)



sns.distplot(train['repeat_play_chance'], bins=rplay_bins, kde=False,

             hist_kws={"alpha": 1})

plt.xlabel('Chance of repeated listen')

plt.ylabel('# of users')

# plt.yscale('log')

# plt.xscale('log')
x_plays = []

y_repeat_chance = []



for i in range(1,train['plays'].max()+1):

    plays_i = train[train['plays']==i]

    count = plays_i['plays'].sum()

    if count > 0:

        x_plays.append(i)

        y_repeat_chance.append(plays_i['repeat_events'].sum() / count)
f,axarray = plt.subplots(1,1,figsize=(15,10))

plt.xlabel('Number of song plays')

plt.ylabel('Chance of repeat listens')

plt.plot(x_plays, y_repeat_chance)
train_basic = train[['msno', 'plays', 'repeat_events', 'repeat_play_chance']].drop_duplicates()

# we create a DF with just the basic info for each user
lang_group = train[['msno', 'language']].groupby(['msno'])

lang_group_nunique = lang_group.agg({"language": pd.Series.nunique})

lang_group_mostfreq = lang_group.agg({"language": lambda x: x.value_counts().index[0]})



lang_group_nunique.reset_index(inplace=True)

lang_group_nunique.columns = list(map(''.join, lang_group_nunique.columns.values))



lang_group_mostfreq.reset_index(inplace=True)

lang_group_mostfreq.columns = list(map(''.join, lang_group_mostfreq.columns.values))



train_lang_nunique = train_basic.merge(lang_group_nunique, on='msno')
y_repeat_chance = []

y_plays = []



for i in range(1,int(lang_group_nunique['language'].max())+1):

    plays_i = train_lang_nunique[train_lang_nunique['language']==i]

    count = plays_i['plays'].sum()

    if count > 0:

        y_plays.append(count)

        y_repeat_chance.append(plays_i['repeat_events'].sum() / count)
fig = plt.figure(figsize=(15, 14)) 

ax1 = plt.subplot(2,1,1)

sns.barplot(x=list(range(1,int(lang_group_nunique['language'].max())+1)),

            y=np.log10(y_plays))

ax1.set_ylabel('log10(# of plays)')



ax2 = plt.subplot(2,1,2)

sns.barplot(x=list(range(1,int(lang_group_nunique['language'].max())+1)),

            y=y_repeat_chance)

ax2.set_ylabel('Chance of repeated listen')



ax2.set_xlabel('# Of languages the users listen to')
lang_group_mostfreq.columns = ['msno', 'main_lang']

train_lang_mostfreq = train.merge(lang_group_mostfreq, on='msno')

train_lang_mostfreq['not_main'] = 0

row_ids = train_lang_mostfreq[train_lang_mostfreq["language"] != train_lang_mostfreq["main_lang"]].index

train_lang_mostfreq['not_main'][row_ids] = 1
train_lang_mostfreq_gb = train_lang_mostfreq[['msno', 'not_main']].groupby(['msno']).agg(['count', 'sum'])



train_lang_mostfreq_gb.reset_index(inplace=True)

train_lang_mostfreq_gb.columns = list(map(''.join, train_lang_mostfreq_gb.columns.values))



train_lang_mostfreq_gb.columns = ['msno', 'plays', 'not_main_plays']  #rename columns

train_lang_mostfreq_gb['not_main_percent'] = train_lang_mostfreq_gb['not_main_plays'] / train_lang_mostfreq_gb['plays']
mostfreq_df = train_basic.merge(train_lang_mostfreq_gb[['msno','not_main_percent']], on='msno')
rplay_bins = np.linspace(-0.01,mostfreq_df['not_main_percent'].max(),50)



labels = list(range(rplay_bins.shape[0]-1))

mostfreq_df['cuts'] = pd.cut(mostfreq_df['not_main_percent'],

                                      bins=rplay_bins, labels=labels)



y_repeat_chance_tc = []

y_plays_tc = []

for i in labels:

    cut_i = mostfreq_df[mostfreq_df['cuts']==i]

    count = cut_i['plays'].sum()

    y_plays_tc.append(count)

    if count != 0:

        y_repeat_chance_tc.append(cut_i['repeat_events'].sum() / count)

    else:

        y_repeat_chance_tc.append(0)

    

fig = plt.figure(figsize=(15, 16)) 



y_plays_tc = [yptc + 1 for yptc in y_plays_tc]  # otherwise we'll get errors when we take the log



ax211 = plt.subplot(2,1,1)

sns.barplot(x=rplay_bins[labels],y=np.log10(y_plays_tc))

ax211.set_ylabel('log10(# of plays)')



ax212 = plt.subplot(2,1,2)

sns.barplot(x=rplay_bins[labels],y=y_repeat_chance_tc)

ax212.set_ylabel('Chance of repeated listen')
plt.figure(figsize=(15,8))

rplay_bins = np.linspace(0,mostfreq_df['not_main_percent'].max(),50)



sns.distplot(mostfreq_df['not_main_percent'], bins=rplay_bins, kde=False,

             hist_kws={"alpha": 1})

# plt.yscale('log')

plt.xlabel('Non-main-language song fraction')

plt.ylabel('# of users')

# plt.xscale('log')
diff_artists = train[['msno', 'artist_name']].groupby(['msno'])

diff_artists = diff_artists.agg({"artist_name": pd.Series.nunique})



diff_artists.reset_index(inplace=True)

diff_artists.columns = ['msno', 'nunique_artists']



diff_artists = train_basic.merge(diff_artists, on='msno')

diff_artists['n_unique_artists_div_plays'] = diff_artists['nunique_artists'] / diff_artists['plays']
plt.figure(figsize=(15,16))

rplay_bins = np.linspace(0,1.,100)

rplay_bins2 = np.logspace(0.9,np.log10(diff_artists['nunique_artists'].max()),100)





ax211 = plt.subplot(2,1,1)

sns.distplot(diff_artists['n_unique_artists_div_plays'], bins=rplay_bins, kde=False,

             hist_kws={"alpha": 1})



ax211.set_xlabel("# of unique artists/ # of user's plays")

ax211.set_ylabel('# of users')

# ax211.set_yscale('log')  # why isn't this working???



ax212 = plt.subplot(2,1,2)



sns.distplot(diff_artists['nunique_artists'], bins=rplay_bins2, kde=False,

             hist_kws={"alpha": 1})



ax212.set_xlabel('# of unique artists')

ax212.set_ylabel('# of users')

# ax212.set_yscale('log')
y_repeat_chance = []

y_plays = []

x_artists = []



for i in range(1,int(diff_artists['nunique_artists'].max())+1):

    plays_i = diff_artists[diff_artists['nunique_artists']==i]

    count = plays_i['plays'].sum()

    if count > 0:

        x_artists.append(i)

        y_plays.append(count)

        y_repeat_chance.append(plays_i['repeat_events'].sum() / count)
fig = plt.figure(figsize=(15, 14)) 

ax1 = plt.subplot(2,1,1)

sns.barplot(x=x_artists,

            y=np.log10(y_plays))

ax1.set_ylabel('log10(# of plays)')



ax2 = plt.subplot(2,1,2)

sns.barplot(x=x_artists,

            y=y_repeat_chance)

ax2.set_ylabel('Chance of repeated listen')



ax2.set_xlabel('# of unique artists')
rplay_bins = np.linspace(-0.01,1.,100)



labels = list(range(rplay_bins.shape[0]-1))

diff_artists['cuts'] = pd.cut(diff_artists['n_unique_artists_div_plays'],

                              bins=rplay_bins, labels=labels)



y_repeat_chance_da = []

y_plays_da = []

for i in labels:

    cut_i = diff_artists[diff_artists['cuts']==i]

    count = cut_i['plays'].sum()

    y_plays_da.append(count)

    if count != 0:

        y_repeat_chance_da.append(cut_i['repeat_events'].sum() / count)

    else:

        y_repeat_chance_da.append(0)

    

fig = plt.figure(figsize=(15, 16)) 



y_plays_da = [ypda + 1 for ypda in y_plays_da]  # otherwise we'll get errors when we take the log



ax211 = plt.subplot(2,1,1)

sns.barplot(x=rplay_bins[labels],y=np.log10(y_plays_da))

ax211.set_ylabel('log10(# of plays)')



ax212 = plt.subplot(2,1,2)

sns.barplot(x=rplay_bins[labels],y=y_repeat_chance_da)

ax212.set_xlabel("# of unique artists/ # of user's plays")

ax212.set_ylabel('Chance of repeated listen')
train['song_length_s'] = train['song_length'] / 1000

sl_gb = train[['msno', 'song_length_s']].groupby(['msno']).agg(['mean', 'std'])



sl_gb.reset_index(inplace=True)

sl_gb.columns = list(map(''.join, sl_gb.columns.values))

sl_gb.columns = ['msno', 'song_length_s_mean', 'song_length_s_std']  #rename columns

sl_gb = train_basic.merge(sl_gb, on='msno')
slay_bins = np.logspace(np.log10(sl_gb['song_length_s_mean'].min()-1),

                        np.log10(sl_gb['song_length_s_mean'].max()+1),100)



labels = list(range(slay_bins.shape[0]-1))

sl_gb['cuts_slm'] = pd.cut(sl_gb['song_length_s_mean'],

                           bins=slay_bins, labels=labels)



y_repeat_chance_sl = []

y_plays_sl = []

y_users_sl = []

for i in labels:

    cut_i = sl_gb[sl_gb['cuts_slm']==i]

    count = cut_i['plays'].sum()

    y_plays_sl.append(count)

    if count != 0:

        y_repeat_chance_sl.append(cut_i['repeat_events'].sum() / count)

    else:

        y_repeat_chance_sl.append(0)

    

fig = plt.figure(figsize=(15, 16)) 



y_plays_sl = [x + 1 for x in y_plays_sl]  # otherwise we'll get errors when we take the log



ax211 = plt.subplot(2,1,1)

sns.barplot(x=slay_bins[labels],y=np.log10(y_plays_sl))

ax211.set_ylabel('log10(# of plays)')



ax212 = plt.subplot(2,1,2)

sns.barplot(x=slay_bins[labels],y=y_repeat_chance_sl)

ax212.set_xlabel("Mean song length")

ax212.set_ylabel('Chance of repeated listen')
slay_bins = np.logspace(np.log10(sl_gb['song_length_s_std'].min()+0.1),

                        np.log10(sl_gb['song_length_s_std'].max()+1),100)



labels = list(range(slay_bins.shape[0]-1))

sl_gb['cuts_slm'] = pd.cut(sl_gb['song_length_s_std'],

                           bins=slay_bins, labels=labels)



y_repeat_chance_sl = []

y_plays_sl = []

y_users_sl = []

for i in labels:

    cut_i = sl_gb[sl_gb['cuts_slm']==i]

    count = cut_i['plays'].sum()

    y_plays_sl.append(count)

    if count != 0:

        y_repeat_chance_sl.append(cut_i['repeat_events'].sum() / count)

    else:

        y_repeat_chance_sl.append(0)

    

fig = plt.figure(figsize=(15, 16)) 



y_plays_sl = [x + 1 for x in y_plays_sl]  # otherwise we'll get errors when we take the log



ax211 = plt.subplot(2,1,1)

sns.barplot(x=slay_bins[labels],y=np.log10(y_plays_sl))

ax211.set_ylabel('log10(# of plays)')



ax212 = plt.subplot(2,1,2)

sns.barplot(x=slay_bins[labels],y=y_repeat_chance_sl)

ax212.set_xlabel("Std of song lengths")

ax212.set_ylabel('Chance of repeated listen')