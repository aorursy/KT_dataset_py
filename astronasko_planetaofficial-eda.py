!pip install transliterate



# Including main libraries

import numpy as np

import pandas as pd

import scipy.stats as stats

import matplotlib.pyplot as plt

import matplotlib.colors

from matplotlib.ticker import AutoMinorLocator

import seaborn as sns

from transliterate import translit

import datetime as dt



def shorten_name(name):

    '''Shortens the artist name for visualisation. Keeps the artist's first name unchanged.'''

    # Splits the input string in words

    name_count = len(name.split())

    

    if (name_count==1 or name=="Desi Slava"):

        return name

    

    # Keeps the first two names only

    forename, surname = name.split()[:2]

    #  Abbreviates the surname

    shortened_name = "{0:} {1:}.".format(forename, surname[0])

    return shortened_name

    

def top_feature_songs(data, n, feature):

    '''Returns the first n songs, sorted by the feature in question.'''

    columns = columns = ['artist_1','artist_2','artist_3','track_name', feature]

    

    out = data.nlargest(n, columns=feature,keep='all')[columns].style.hide_index()

    

    display(out)



# Load the entire dataset in data

data = pd.read_csv("/kaggle/input/payner/payner.csv")
# Get year of release

data['year'] = data.datetime.apply(lambda x: x[0:4]).astype(int)



# Disregard aforementioned columns

data = data.drop(

    columns=[

        'track_id',

        'popularity',

        'mode',

        'key',

        'time_signature',

        'instrumentalness',

        'liveness',

        'datetime'

    ])



# Reorder remaining columns

new_cols = ['track_name', 'year', 'artist_1', 'artist_2', 'artist_3',

            'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',

            'valence', 'tempo', 'duration']



data = data[new_cols]



#Shorten artist names

data.artist_1 = data.artist_1.apply(shorten_name)

data.artist_2 = data.artist_2.apply(shorten_name)

data.artist_3 = data.artist_3.apply(shorten_name)



# Transform duration in minutes

data.duration = data.duration.apply(lambda x: x/1000/60)
first_second_third = pd.concat([data.artist_1,data.artist_2,data.artist_3])



data_artists = pd.DataFrame({

    'artist_1': data.artist_1.value_counts(dropna=False),

    'artist_2': data.artist_2.value_counts(dropna=False),

    'artist_3': data.artist_3.value_counts(dropna=False)

}).fillna(0)



key = first_second_third.value_counts().index[1:16].tolist()



data_artists.reindex(key).plot(

    kind='bar',

    stacked=True,

    figsize=(12, 4),

    color=['gold','darkgray','saddlebrown'],

    edgecolor="black",

    linewidth=0.5,

    zorder=2)



plt.title("Most prevalent artists, PlanetaOfficial, 2014-2019",fontsize=14)

plt.legend(labels=['First', 'Second', 'Third'], title="Order in track", loc='best')

plt.grid(axis='y', linewidth=0.5, zorder=0)

plt.xticks(rotation='horizontal', wrap=True)

plt.ylim(0,30)

plt.ylabel("Track count")

plt.tight_layout()
# Artists by solo songs - important for correlation matrices!

# Get only solo tracks

data_solo = data[(data.artist_2=='None')&(data.artist_3=='None')].drop(columns=['artist_2','artist_3'])



# Get all authors by solo track count

count_solo = pd.DataFrame({

    'solist': data_solo.artist_1.value_counts(dropna=False),

})



# Get names of first 10 authors by solo track count

key_solo = data_solo.artist_1.value_counts().index[0:10].tolist()



# Plot only first 10 authors from all authors

count_solo.reindex(key_solo).plot(

    kind='bar',

    figsize=(12, 4),

    color='tab:blue',

    edgecolor="black",

    linewidth=0.5,

    zorder=2)



plt.title("Top authors by solo tracks, Planeta Payner, 2014-2019",fontsize=14)

plt.legend(labels=['Solo tracks'], loc='best')

plt.xticks(rotation='horizontal', wrap=True)

plt.grid(axis='y', linewidth=0.5, zorder=0)

plt.tight_layout()
# Has this artist got at least 3 tracks in which they are mentioned first?

# Preparing a boolean mask of the condition

mask = data.artist_1.map(data.artist_1.value_counts())  >= 3

# Applying this boolean mask to data

data = data[mask]
top_feature_songs(data, n=5, feature='danceability')
top_feature_songs(data, n=5, feature='energy')
top_feature_songs(data, n=5, feature='loudness')
top_feature_songs(data, n=5, feature='speechiness')
top_feature_songs(data, n=5, feature='acousticness')
top_feature_songs(data, n=5, feature='valence')
top_feature_songs(data, n=5, feature='tempo')
f, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)



mask = np.triu(np.ones_like(data.drop(columns=['year']).corr(), dtype=np.bool))



ax_pearson = sns.heatmap(

    data.drop(columns=['year']).corr(method='pearson'),

    mask=mask,

    vmax=1,

    vmin=-1,

    square=True,

    annot=True,

    fmt=".2f",

    cbar=False,

    cmap="RdBu_r",

    ax=axes[0])



ax_pearson.set_title('Pearson',fontsize=14)



ax_spearman = sns.heatmap(

    data.drop(columns=['year']).corr(method='spearman'),

    mask=mask,

    vmax=1,

    vmin=-1,

    square=True,

    annot=True,

    fmt=".2f",

    cbar=False,

    cmap="RdBu_r",

    ax=axes[1])



ax_spearman.set_title('Spearman',fontsize=14)



plt.tight_layout()

plt.suptitle("Features correlation of tracks, PlanetaOfficial, 2014-2019", fontsize=18, y=1.10)

plt.subplots_adjust(hspace = 0.10)
plt.figure(figsize=(9, 6))

plt.title("Co-dependency of energy and loudness, PlanetaOfficial, 2014-2019", fontsize=14, y=1.04)

plt.xlabel("Energy coefficient")

plt.ylabel("Loudness (dB)")

sns.scatterplot(x="energy", y="loudness", data=data, alpha=0.3);

sns.regplot(x="energy", y="loudness", data=data, scatter=False);
f, axes = plt.subplots(3, 2, figsize=(12, 14), sharex=True)

mask = np.triu(np.ones_like(data_solo.corr(), dtype=np.bool))



for i in range(3):

    

    artist = key_solo[i]

    

    data_artist = data_solo.loc[data_solo.artist_1 == artist].drop(columns=['track_name','year', 'artist_1'])

    data_artist = data_artist[(np.abs(stats.zscore(data_artist)) < 3).all(axis=1)]

    

    mask = np.triu(np.ones_like(data_artist.corr(), dtype=np.bool))

    # Pearson

    sns.heatmap(

        data=data_artist.corr(method='pearson'),

        mask=mask,

        vmax=1,

        vmin=-1,

        ax=axes[i, 0],

        square=True,

        annot=True,

        fmt=".2f",

        cbar=False,

        cmap="RdBu_r"

    )

    # Spearman

    sns.heatmap(

        data=data_artist.corr(method='spearman'),

        mask=mask,

        vmax=1,

        vmin=-1,

        ax=axes[i, 1],

        square=True,

        annot=True,

        fmt=".2f",

        cbar=False,

        cmap="RdBu_r"

    )

    # Left-hand-side labels

    pop_size = count_solo.solist[i]

    sam_size = len(data_artist)

    

    axes[i, 0].set_ylabel(

        '{0:}, n={1:}\n({2:} solo tracks)'.format(artist, sam_size, pop_size),

        fontsize=14,

        rotation=0,

        labelpad=80,

        va='center',

        linespacing=1.4

        

    )

    

    axes[i,0].set_title('Pearson', fontsize=14)

    axes[i,1].set_title('Spearman', fontsize=14)



plt.suptitle("Features correlation of artists, PlanetaOfficial, 2014-2019", fontsize=18, y=1.04)

plt.subplots_adjust(hspace = 0.15)

plt.tight_layout()
f, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)



plt.suptitle("Tendencies of artists, PlanetaOfficial, 2014-2019", fontsize=16, y=1.05)

plt.xlim(2.5, 5)

plt.ylim(0, 0.3)

plt.xticks(

    ticks = np.linspace(2.5,5,6),

    labels = ['2:30','3:00','3:30','4:00','4:30','5:00'])



# DZHENA

axes[0].set_title("Solo tracks by Dzhena")

# All dots

sns.scatterplot(

    x="duration",

    y="speechiness",

    data=data,

    alpha=0.15,

    color='gray',

    ax=axes[0])

# Dzhena scatter

sns.scatterplot(

    x="duration",

    y="speechiness",

    data = data_solo[data_solo['artist_1']=="Dzhena"],

    color="tab:blue",

    ax=axes[0])

# Dzhena reg

sns.regplot(

    x="duration",

    y="speechiness",

    data=data_solo[data_solo['artist_1']=="Dzhena"],

    color="tab:blue",

    scatter=False,

    ax=axes[0])



# PRESLAVA

axes[1].set_title("Solo tracks by Preslava")

# All dots

sns.scatterplot(

    x="duration",

    y="speechiness",

    data=data,

    alpha=0.15,

    color='gray',

    ax=axes[1])

# Scatter

sns.scatterplot(

    x="duration",

    y="speechiness",

    data = data_solo[data_solo['artist_1']=="Preslava"],

    color="tab:orange",

    ax=axes[1])

# Reg

sns.regplot(

    x="duration",

    y="speechiness",

    data=data_solo[data_solo['artist_1']=="Preslava"],

    color="tab:orange",

    scatter=False,

    ax=axes[1]);
# Three-fold binning of data

data_old = data[data.year.isin([2014,2015])]

data_med = data[data.year.isin([2016,2017])]

data_new = data[data.year.isin([2018,2019])]



# Providing axis limits for features

feature_limits = {

    'danceability': (0.4, 0.9),

    'energy': (0.6, 1),

    'loudness': (-8, 0),

    'speechiness': (0, 0.3),

    'acousticness': (0, 0.3),

    'liveness': (0, 0.6),

    'valence': (0.2, 1),

    'tempo': (50, 250)

}
f, axes = plt.subplots(3, 1, figsize=(8, 8), sharex='all', sharey='all')



bins = np.linspace(2.5,5,31)



row = 0

for data in [data_old, data_med, data_new]:



    ax = sns.distplot(

    data.duration,

    norm_hist=True,

    ax=axes[row],

    color='tab:blue',

    bins=bins)

        

    ax.grid(axis='y', which='major', color='k', linestyle='-', alpha=0.2, zorder=100)

    ax.grid(axis='x', which='major', color='k', linestyle='-', alpha=0.2, zorder=100)

    

    ax.set_xlim(2.5, 5)

    ax.set_xticklabels(['2:30','3:00','3:30','4:00','4:30','5:00'])

    ax.minorticks_on()

    ax.xaxis.set_minor_locator(AutoMinorLocator(6))

    

    ax.set_xlabel("Track duration")

    ax.set_ylabel("KDE")



    row += 1

    



axes[0].set_title(r"2014-2015 $(n=131)$")

axes[1].set_title(r"2016-2017 $(n=122)$")

axes[2].set_title(r"2018-2019 $(n=104)$")



plt.suptitle("Tempo of tracks by PlanetaOfficial, 2014-2019", fontsize=14, y=1.02)

plt.subplots_adjust(top=0.99)

plt.grid(axis='x', linewidth=0.5, zorder=0)

plt.tight_layout()
f, axes = plt.subplots(7, 3, figsize=(9, 21),sharey='row', sharex='row')



row = 0



for feature in data_old.columns[5:-1]: #['danceability',...'duration']

    

    ax_old = sns.kdeplot(

        data_old['duration'],

        data_old[feature],

        shade=True,

        ax=axes[row,0],

        shade_lowest=False, 

        cmap="YlOrRd")



    ax_med= sns.kdeplot(

        data_med['duration'],

        data_med[feature],

        shade=True,

        ax=axes[row,1],

        shade_lowest=False, 

        cmap="YlOrRd")

    

    ax_new = sns.kdeplot(

        data_new['duration'],

        data_new[feature],

        shade=True,

        ax=axes[row,2],

        shade_lowest=False, 

        cmap="YlOrRd")

    

    ax_old.set_xlim(2.5, 5)

    ax_old.set_xticklabels(['2:30','3:00','3:30','4:00','4:30','5:00'])

    

    ax_old.set_ylim(*feature_limits[feature])

        

    axes[0,0].set_title(r"2014-2015 $(n=131)$")

    axes[0,1].set_title(r"2016-2017 $(n=122)$")

    axes[0,2].set_title(r"2018-2019 $(n=104)$")

    

    row += 1



plt.suptitle("Feature heatmaps over time, PlanetaOfficial, 2014-2019", fontsize=16, y=1.02)

plt.subplots_adjust(top=0.95)

plt.tight_layout()
f, axes = plt.subplots(1, 3, figsize=(12, 4),sharey='row', sharex=True)



ax_old = sns.kdeplot(

    data_old['duration'],

    data_old['tempo'],

    shade=True,

    ax=axes[0],

    shade_lowest=False, 

    cmap="YlOrRd")



ax_med= sns.kdeplot(

    data_med['duration'],

    data_med['tempo'],

    shade=True,

    ax=axes[1],

    shade_lowest=False, 

    cmap="YlOrRd")



ax_new = sns.kdeplot(

    data_new['duration'],

    data_new['tempo'],

    shade=True,

    ax=axes[2],

    shade_lowest=False, 

    cmap="YlOrRd")



ax_old.set_xlim(2.5, 5)

ax_old.set_xticklabels(['2:30','3:00','3:30','4:00','4:30','5:00'])



ax_old.set_ylim(*feature_limits['tempo'])



axes[0].set_title(r"2014-2015 $(n=131)$")

axes[1].set_title(r"2016-2017 $(n=122)$")

axes[2].set_title(r"2018-2019 $(n=104)$")



plt.suptitle("Duration-tempo over time, PlanetaOfficial, 2014-2019", fontsize=14, y=1.02)

plt.subplots_adjust(top=0.95)

plt.tight_layout()
f, axes = plt.subplots(1, 3, figsize=(12, 4),sharey='row', sharex=True)



ax_old = sns.kdeplot(

    data_old['duration'],

    data_old['loudness'],

    shade=True,

    ax=axes[0],

    shade_lowest=False, 

    cmap="YlOrRd")



ax_med= sns.kdeplot(

    data_med['duration'],

    data_med['loudness'],

    shade=True,

    ax=axes[1],

    shade_lowest=False, 

    cmap="YlOrRd")



ax_new = sns.kdeplot(

    data_new['duration'],

    data_new['loudness'],

    shade=True,

    ax=axes[2],

    shade_lowest=False, 

    cmap="YlOrRd")



ax_old.set_xlim(2.5, 5)

ax_old.set_xticklabels(['2:30','3:00','3:30','4:00','4:30','5:00'])



ax_old.set_ylim(*feature_limits['loudness'])



axes[0].set_title(r"2014-2015 $(n=131)$")

axes[1].set_title(r"2016-2017 $(n=122)$")

axes[2].set_title(r"2018-2019 $(n=104)$")



plt.suptitle("Duration-loudness over time, PlanetaOfficial, 2014-2019", fontsize=14, y=1.02)

plt.subplots_adjust(top=0.95)

plt.tight_layout()