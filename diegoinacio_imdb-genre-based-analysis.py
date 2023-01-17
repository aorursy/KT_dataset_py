%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../input/tmdb_5000_movies.csv')
df.head().T
print('{:>28}'.format('entries from dataset:'), df.shape[0])
df = df.drop_duplicates(['original_title'])
print('{:>28}'.format('entries without duplicated:'), df.shape[0])
df_clean = df[['budget', 'genres', 'release_date', 'revenue']].dropna()
print('{:>28}'.format('entries from cleaned data:'), df_clean.shape[0])
df_clean.head(10).T
df_genre = pd.DataFrame(columns = ['genre', 'cgenres', 'budget', 'revenue', 'day', 'month', 'year'])

def dataPrep(row):
    global df_genre
    d = {}
    genres = np.array([g['name'] for g in eval(row['genres'])])
    n = genres.size
    d['budget'] = [row['budget']]*n
    d['revenue'] = [row['revenue']]*n
    d.update(zip(('year', 'month', 'day'), map(int, row['release_date'].split('-'))))
    d['genre'], d['cgenres'] = [], []
    for genre in genres:
        d['genre'].append(genre)
        d['cgenres'].append(genres[genres != genre])
    df_genre = df_genre.append(pd.DataFrame(d), ignore_index=True, sort=True)

df_clean.apply(dataPrep, axis=1)
df_genre = df_genre[['genre', 'budget', 'revenue', 'day', 'month', 'year', 'cgenres']]
df_genre = df_genre.infer_objects()
df_clean[['genres', 'release_date']].head(2) # before data preparation
df_genre[['genre', 'cgenres', 'year']].head(7) # after data preparation
print('{:>32}'.format('entries before data preparation:'), df_clean.shape[0])
print('{:>32}'.format('entries after data preparation:'), df_genre.shape[0])
df_genre.head(10).T
genre_count = df_genre['genre'].value_counts().sort_index()
df_gCount = pd.DataFrame({'genre': genre_count.index, 'count': genre_count.values})
f, ax = plt.subplots(figsize=(23, 9))
sns.barplot(x = 'count', y = 'genre', data=df_gCount)
ax.set_title('.: occurences per genre :.')
ax.set_xlabel('occurrences')
ax.set_ylabel('genres')
plt.show()
genre_year = df_genre.groupby(['genre', 'year']).mean().sort_index()
df_gyBudget = genre_year.pivot_table(index=['genre'], columns=['year'], values='budget', aggfunc=np.mean)
df_gyGross = genre_year.pivot_table(index=['genre'], columns=['year'], values='revenue', aggfunc=np.mean)
f, [axA, axB] = plt.subplots(figsize=(27, 11), nrows=2)
cmap = sns.cubehelix_palette(start=1.5, rot=1.5, as_cmap=True)
sns.heatmap(df_gyBudget, xticklabels=3, cmap=cmap, linewidths=0.05, ax=axA)
sns.heatmap(df_gyGross, xticklabels=3, cmap=cmap, linewidths=0.05, ax=axB)
axA.set_title('.: budget x genre x year :.')
axA.set_xlabel('years')
axA.set_ylabel('genres')
axB.set_title('.: revenue x genre x year :.')
axB.set_xlabel('years')
axB.set_ylabel('genres')
plt.show()
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
f, ax = plt.subplots(figsize = (23, 13))

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