# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import json



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import Imputer



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb_5000_credits.csv')
del credits['title']

df = pd.concat([movies, credits], axis=1)
df.head()
def load_tmdb_movies(path):

    df = pd.read_csv(path)

    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())

    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df



def load_tmdb_credits(path):

    df = pd.read_csv(path)

    json_columns = ['cast', 'crew']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df



def pipe_flatten_names(keywords):

    return '|'.join([x['name'] for x in keywords])
credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")

movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")
del credits['title']

df = pd.concat([movies, credits], axis=1)
df['genres'] = df['genres'].apply(pipe_flatten_names)
liste_genres = set()

for s in df['genres'].str.split('|'):

    liste_genres = set().union(s, liste_genres)

liste_genres = list(liste_genres)

liste_genres.remove('')
df_reduced = df[['title','vote_average','release_date','runtime','budget','revenue']].reset_index(drop=True)
df_reduced.head()
for genre in liste_genres:

    df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)

df_reduced[:5]
plt.rc('font', weight='bold')

f, ax = plt.subplots(figsize=(5,5))

genre_count = []

for genre in liste_genres:

    genre_count.append([genre, df_reduced[genre].values.sum()])

genre_count.sort(key = lambda x:x[1], reverse = True)

labels, sizes = zip(*genre_count)

labels_selected = [n if v > sum(sizes) * 0.01 else '' for n, v in genre_count]

ax.pie(sizes, labels=labels_selected,

      autopct = lambda x:'{:2.0f}%'.format(x) if x>1 else '',

      shadow = False, startangle=0)

ax.axis('equal')

plt.tight_layout()
mean_per_genre = pd.DataFrame(liste_genres)
newArray = []*len(liste_genres)

for genre in liste_genres:

    newArray.append(df_reduced.groupby(genre, as_index=True)['vote_average'].mean())

newArray2 = []*len(liste_genres)

for i in range(len(liste_genres)):

    newArray2.append(newArray[i][1])



mean_per_genre['mean_votes_average']=newArray2
newArray = []*len(liste_genres)

for genre in liste_genres:

    newArray.append(df_reduced.groupby(genre, as_index=True)['budget'].mean())

newArray2 = []*len(liste_genres)

for i in range(len(liste_genres)):

    newArray2.append(newArray[i][1])



mean_per_genre['mean_budget']=newArray2
newArray = []*len(liste_genres)

for genre in liste_genres:

    newArray.append(df_reduced.groupby(genre, as_index=True)['revenue'].mean())

newArray2 = []*len(liste_genres)

for i in range(len(liste_genres)):

    newArray2.append(newArray[i][1])



mean_per_genre['mean_revenue']=newArray2
mean_per_genre['profit'] = mean_per_genre['mean_revenue']-mean_per_genre['mean_budget']
mean_per_genre
mean_per_genre['mean_votes_average'].plot.barh()
from datetime import datetime

#list(map(datetime.year, df_reduced["release_date"]))

t = df_reduced['release_date']

t = pd.to_datetime(t)

t = t.dt.year

df_reduced['release_year'] = t
df_list = []*len(liste_genres)

for genre in liste_genres:

    df_list.append(df_reduced.groupby([genre,'release_year']).mean().reset_index())
df_per_genre = []*len(liste_genres)

for i in range(len(df_list)):

    df_per_genre.append(df_list[i][df_list[i].ix[:,0] == 1])
columns = range(1988,2018)

budget_genre = pd.DataFrame( columns = columns)

budget_genre
for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'budget', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    budget_genre.loc[liste_genres.index(genre)]=temp

budget_genre['genre']=liste_genres
budget_genre.index = budget_genre['genre']

budget_genre
columns = range(1988,2018)

revenue_genre = pd.DataFrame( columns = columns)

revenue_genre
for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'revenue', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    revenue_genre.loc[liste_genres.index(genre)]=temp

revenue_genre['genre']=liste_genres
revenue_genre.index = revenue_genre['genre']

revenue_genre
columns = range(1988,2018)

vote_avg_genre = pd.DataFrame( columns = columns)

vote_avg_genre
for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'vote_average', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    vote_avg_genre.loc[liste_genres.index(genre)]=temp

vote_avg_genre['genre']=liste_genres
vote_avg_genre.index = vote_avg_genre['genre']

vote_avg_genre
len(budget_genre)
#f, [axA, axB] = plt.subplots(figsize = (9, 9), nrows = 2)

fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(budget_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)

#axA.set_ylabel('genres')

#df.ix[:,0:2]
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(revenue_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(vote_avg_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
temp = budget_genre

temp


temp[2013]=temp[2013].replace(2.550000e+08, 0)
temp
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(temp.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
df_genre = pd.DataFrame(columns = ['genre', 'cgenres', 'budget', 'gross', 'year'])
from datetime import datetime

#list(map(datetime.year, df_reduced["release_date"]))

t = df['release_date']

t = pd.to_datetime(t)

t = t.dt.year

df_genre['release_year'] = t
colnames = ['budget', 'genres', 'revenue']

df_clean = df[colnames]
df_clean['release_year'] = t
df_clean = df_clean.dropna()

df_genre = df_genre.dropna()
df_clean.head()
def genreRemap(row):

    global df_genre

    d = {}

    genres = np.array(row['genres'].split('|'))

    n = genres.size

    d['budget'] = [row['budget']]*n

    d['revenue'] = [row['revenue']]*n

    d['year'] = [row['release_year']]*n

    d['genre'], d['cgenres'] = [], []

    for genre in genres:

        d['genre'].append(genre)

        d['cgenres'].append(genres[genres != genre])

    df_genre = df_genre.append(pd.DataFrame(d), ignore_index = True)



df_clean.apply(genreRemap, axis = 1)

df_genre['year'] = df_genre['year'].astype(np.int16)

df_genre = df_genre[['genre', 'budget', 'gross', 'year', 'cgenres']]
####################

# make connections #

####################

d_genre = {}

def connect(row):

    global d_genre

    genre = row['genre']

    cgenres = row['cgenres']

    if genre not in d_genre:

        d_cgenres = dict(zip(cgenres, [1]*len(cgenres)))

        d_genre[genre] = d_cgenres

    else:

        for cgenre in cgenres:

            if cgenre not in d_genre[genre]:

                d_genre[genre][cgenre] = 1

            else:

                d_genre[genre][cgenre] += 1

                

df_genre.apply(connect, axis = 1)

l_genre = list(d_genre.keys())

l_genre.sort()

###########################

# find largest connection #

###########################

cmax = 0

for key in d_genre:

    for e in d_genre[key]:

        if d_genre[key][e] > cmax:

            cmax = d_genre[key][e]

#########################

# visualize connections #

#########################

from matplotlib.path import Path

import matplotlib.patches as patches

from matplotlib import cm

color = cm.get_cmap('rainbow')

f, ax = plt.subplots(figsize = (7, 9))



codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]



X, Y = 1, 1

wmin, wmax = 1, 32

amin, amax = 0.1, 0.25

getPy = lambda x: Y*(1 - x/len(l_genre))

for i, genre in enumerate(l_genre):

    yo = getPy(i)

    ax.text(0, yo, genre, ha = 'right')

    ax.text(X, yo, genre, ha = 'left')

    for cgenre in d_genre[genre]:

        yi = getPy(l_genre.index(cgenre))

        verts = [(0.0, yo), (X/4, yo), (2*X/4, yi), (X, yi)]

        path = Path(verts, codes)

        r, g, b, a = color(i/len(l_genre))

        width = wmin + wmax*d_genre[genre][cgenre]/cmax

        alpha = amin + amax*(1 - d_genre[genre][cgenre]/cmax)

        patch = patches.PathPatch(path, facecolor = 'none', edgecolor = (r, g, b), lw = width, alpha = alpha)

        ax.add_patch(patch)



ax.grid(False)

ax.set_xlim(0.0, X)

ax.set_ylim(0.0, Y + 1/len(l_genre))

ax.set_yticklabels([])

ax.set_xticklabels([])

plt.show()