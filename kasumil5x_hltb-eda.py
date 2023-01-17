import re

import pandas as pd

import matplotlib.pyplot as plt

%config InlineBackend.figure_formats = ['svg']

%matplotlib inline

plt.style.use('seaborn-ticks')

import seaborn as sns
games = pd.read_csv('../input/all-games-processed.csv', index_col='id')

completions = pd.read_csv('../input/all-completions.csv', index_col='id')
games.isnull().sum()
games[games.title.isna()]
# first one

games.loc[19962, ['title', 'platforms']] = ['N.U.D.E.@ Natural Ultimate Digital Experiment', 'Xbox']



# second one

games.loc[38799, ['title', 'platforms', 'release_jp']] = ['Brut@l', 'PlayStation 4, PC', 'October 20, 2016']
completions.isnull().sum()
# Replace missing `platform` entries with an empty string for concat purposes.

completions['platform'] = completions.platform.fillna('')



# `games` identifiers where platform is null AND there is at least one valid completion entry (not all games will have entries).

missing_platform_ids = games[games.platforms.isnull()].index.intersection(completions.index)



# Create a comma-space separated list of all unique platforms (there may be multiple entries and for different platforms).

replacement_platforms = pd.DataFrame(completions.loc[missing_platform_ids].groupby('id').platform.unique().transform(lambda x: ', '.join(x)))

# Drop any that resulted in an empty string.

replacement_platforms = replacement_platforms[replacement_platforms.platform != '']

# Rename column to match that in `games` for a proper merge.

replacement_platforms.columns = ['platforms']



# Use `update()` to replace NaN entries with any matches.

games.update(replacement_platforms)



# 15021 is from above output. If this changes, the data changed!

print('Reduced `game.platform` NaN values from {} to {}!'.format(15021, games.platforms.isnull().sum()))
games['developers'] = games.developers.fillna('')

games['publishers'] = games.publishers.fillna('')

games['platforms'] = games.platforms.fillna('')

games['genres'] = games.genres.fillna('')
games['developers'] = games.developers.apply(lambda x: re.sub(r',\s?$', '', x))

games['publishers'] = games.publishers.apply(lambda x: re.sub(r',\s?$', '', x))

games['platforms'] = games.platforms.apply(lambda x: re.sub(r',\s?$', '', x))

games['genres'] = games.genres.apply(lambda x: re.sub(r',\s?$', '', x))
games['developers'] = games['developers'].str.replace(r'\s?\(.*?\)', '')
games.loc[47947, 'developers'] = 'quickdraw studios'
# `http://www.cobramobile.com/` and `http://prinnies.com/`

games = games[~games.genres.str.contains('http', na=False)]



# Sanic joke.

games = games[games.genres != 'ANAZIGN']
games['developers'] = games.developers.str.lower()

games['publishers'] = games.publishers.str.lower()

games['platforms'] = games.platforms.str.lower()

games['genres'] = games.genres.str.lower()
# Helper function to split a column by ', ' and return all unique values.

def unique_from_split(col):

    tmp = pd.Series([item for sublist in games[col].str.split(', ') for item in sublist]).unique().tolist()

    tmp.remove('')

    return tmp

#end
games_lookup = {}



# i = 0

for idx, row in games.iterrows():

    # split devs and remove empty strings

    devs = row.developers.split(', ')

    if '' in devs:

        devs.remove('')    

    

    # publishers

    pubs = row.publishers.split(', ')

    if '' in pubs:

        pubs.remove('')

        

    # platforms

    plats = row.platforms.split(', ')

    if '' in plats:

        plats.remove('')

        

    # genres

    genres = row.genres.split(', ')

    if '' in genres:

        genres.remove('')

    

    games_lookup[idx] = {

        'developers': devs,

        'publishers': pubs,

        'platforms': plats,

        'genres': genres

    }

#end
%%time



# developers

developers_lookup = {

}

for dev in unique_from_split('developers'):

    developers_lookup[dev] = [i for i in games_lookup.keys() if dev in games_lookup[i]['developers']]

    

# publishers

publishers_lookup = {

}

for pub in unique_from_split('publishers'):

    publishers_lookup[pub] = [i for i in games_lookup.keys() if pub in games_lookup[i]['publishers']]



# platforms

platforms_lookup = {

}

for plat in unique_from_split('platforms'):

    platforms_lookup[plat] = [i for i in games_lookup.keys() if plat in games_lookup[i]['platforms']]



# genres

genres_lookup = {

}

for gen in unique_from_split('genres'):

    genres_lookup[gen] = [i for i in games_lookup.keys() if gen in games_lookup[i]['genres']]
count_per_platform = pd.DataFrame(

    [(key, len(val)) for key, val in platforms_lookup.items()],

    columns=['Platform', 'Count']

)



g = sns.barplot(data=count_per_platform.sort_values('Count', ascending=False).head(10), x='Platform', y='Count')

sns.despine()

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.set_title('Total games per platform')

plt.show()
count_per_developer = pd.DataFrame(

    [(key, len(val)) for key, val in developers_lookup.items()],

    columns=['Developer', 'Count']

)



g = sns.barplot(data=count_per_developer.sort_values('Count', ascending=False).head(10), x='Developer', y='Count')

sns.despine()

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.set_title('Total games per developer')

plt.show()
count_per_publisher = pd.DataFrame(

    [(key, len(val)) for key, val in publishers_lookup.items()],

    columns=['Publisher', 'Count']

)



g = sns.barplot(data=count_per_publisher.sort_values('Count', ascending=False).head(10), x='Publisher', y='Count')

sns.despine()

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.set_title('Total games per publisher')

plt.show()
count_per_genre = pd.DataFrame(

    [(key, len(val)) for key, val in genres_lookup.items()],

    columns=['Genre', 'Count']

)



g = sns.barplot(data=count_per_genre.sort_values('Count', ascending=False).head(10), x='Genre', y='Count')

sns.despine()

g.set_xticklabels(g.get_xticklabels(), rotation=45)

g.set_title('Total games per genre')

plt.show()
# debug for checking values when writing

games[games.genres.str.contains('action')][['title', 'platforms']].values