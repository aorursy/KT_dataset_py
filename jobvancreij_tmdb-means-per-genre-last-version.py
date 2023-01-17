import json



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 





import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



movies = pd.read_csv('../input/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb_5000_credits.csv')



del credits['title']

df = pd.concat([movies, credits], axis=1)

# Any results you write to the current directory are saved as output.
str_list = [] # empty list to contain columns with strings

for colname, colvalue in movies.iteritems():

    if type(colvalue[1]) == str:

        str_list.append(colname)

#Get to the numeric columns by inversion

num_list = movies.columns.difference(str_list)

#We can create a new data frame containing just the numbers:

movie_num = movies[num_list]

#There still exist NaN values, which we have to get rid of:



movie_num = movie_num.fillna(value=0, axis=1)

#We standardise the data with sklearn's StandardScaler

X = movie_num.values

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)



X = movie_num.values

# Data Normalization

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components = 4)

Y_sklearn = sklearn_pca.fit_transform(X_std)



pca = PCA(n_components=7)

x_7d = pca.fit_transform(X_std)



pca4 = PCA(n_components=4)

x_4d = pca.fit_transform(X_std)



#Set a 3 KMeans clustering

kmeans = KMeans(n_clusters = 3)



#Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_7d)



#Define our own color map

LABEL_COLOR_MAP = {0:'r', 1: 'g', 2: 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(x_7d[:,0],x_7d[:,2], c= label_color, alpha=0.5) 

plt.show()



#Set a 3 KMeans clustering

kmeans = KMeans(n_clusters = 3)



#Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_4d)



#Define our own color map

LABEL_COLOR_MAP = {0:'r', 1: 'g', 2: 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(x_4d[:,0],x_4d[:,2], c= label_color, alpha=0.5) 

plt.show()
# Create a temp dataframe from our PCA projection data "x_9d"

df = pd.DataFrame(x_4d)

df = df[[0,1,2]] # only want to visualise relationships between first 3 projections

df['X_cluster'] = X_clustered



# Call Seaborn's pairplot to visualize our KMeans clustering on the PCA projected data

sns.pairplot(df, hue='X_cluster', palette= 'Dark2', diag_kind='kde',size=1.85)

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



for genre in liste_genres:

    df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)

df_reduced[:5]



df_reduced.head()
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



#Mean votes average

newArray = []*len(liste_genres)

for genre in liste_genres:

    newArray.append(df_reduced.groupby(genre, as_index=True)['vote_average'].mean())

newArray2 = []*len(liste_genres)

for i in range(len(liste_genres)):

    newArray2.append(newArray[i][1])



mean_per_genre['mean_votes_average']=newArray2



#Mean budget

newArray = []*len(liste_genres)

for genre in liste_genres:

    newArray.append(df_reduced.groupby(genre, as_index=True)['budget'].mean())

newArray2 = []*len(liste_genres)

for i in range(len(liste_genres)):

    newArray2.append(newArray[i][1])



mean_per_genre['mean_budget']=newArray2



#Mean revenue 



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



mean_per_genre['mean_budget'].plot.barh()
mean_per_genre['mean_revenue'].plot.barh()

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

# Budget

columns = range(1988,2018)

budget_genre = pd.DataFrame( columns = columns)

budget_genre

for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'budget', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    budget_genre.loc[liste_genres.index(genre)]=temp

budget_genre['genre']=liste_genres



# Revenue 



columns = range(1988,2018)

revenue_genre = pd.DataFrame( columns = columns)



for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'revenue', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    revenue_genre.loc[liste_genres.index(genre)]=temp

revenue_genre['genre']=liste_genres



# Vote average 

columns = range(1988,2018)

vote_avg_genre = pd.DataFrame( columns = columns)

vote_avg_genre['genre'] = liste_genres



for genre in liste_genres:

    temp=(df_per_genre[liste_genres.index(genre)].pivot_table(index = genre, columns = 'release_year', values = 'vote_average', aggfunc = np.mean))

    temp = temp[temp.columns[-30:]].loc[1]

    vote_avg_genre.loc[liste_genres.index(genre)]=temp

vote_avg_genre['genre']=liste_genres







budget_genre.index = budget_genre['genre']

budget_genre
revenue_genre.index = revenue_genre['genre']

revenue_genre

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