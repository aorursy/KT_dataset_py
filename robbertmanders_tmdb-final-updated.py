# Package import

from sklearn.preprocessing import Imputer

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 

from sklearn.neighbors import KNeighborsClassifier

import nltk

from nltk.corpus import wordnet

PS = nltk.stem.PorterStemmer()

import matplotlib.pyplot as plt

import plotly.offline as pyo

pyo.init_notebook_mode()

from plotly.graph_objs import *

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

from pandas.plotting import scatter_matrix

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



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

df_new = pd.concat([movies, credits], axis=1)



df_new['genres'] = df_new['genres'].apply(pipe_flatten_names)

df_new['keywords'] = df_new['keywords'].apply(pipe_flatten_names)
# Columns that existed in the IMDB version of the dataset and are gone.

LOST_COLUMNS = [

    'actor_1_facebook_likes',

    'actor_2_facebook_likes',

    'actor_3_facebook_likes',

    'aspect_ratio',

    'cast_total_facebook_likes',

    'color',

    'content_rating',

    'director_facebook_likes',

    'facenumber_in_poster',

    'movie_facebook_likes',

    'movie_imdb_link',

    'num_critic_for_reviews',

    'num_user_for_reviews'

                ]



# Columns in TMDb that had direct equivalents in the IMDB version. 

# These columns can be used with old kernels just by changing the names

TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {

    'budget': 'budget',

    'genres': 'genres',

    'revenue': 'gross',

    'title': 'movie_title',

    'runtime': 'duration',

    'original_language': 'language',  # it's possible that spoken_languages would be a better match

    'keywords': 'plot_keywords',

    'vote_count': 'num_voted_users',

                                         }



IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}





def safe_access(container, index_values):

    # return a missing value rather than an error upon indexing/key failure

    result = container

    try:

        for idx in index_values:

            result = result[idx]

        return result

    except IndexError or KeyError:

        return pd.np.nan





def get_director(crew_data):

    directors = [x['name'] for x in crew_data if x['job'] == 'Director']

    return safe_access(directors, [0])





def pipe_flatten_names(keywords):

    return '|'.join([x['name'] for x in keywords])





def convert_to_original_format(movies, credits):

    # Converts TMDb data to make it as compatible as possible with kernels built on the original version of the data.

    tmdb_movies = movies.copy()

    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)

    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)

    # I'm assuming that the first production country is equivalent, but have not been able to validate this

    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['director_name'] = credits['crew'].apply(get_director)

    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))

    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))

    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))

    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)

    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)

    return tmdb_movies



credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")

movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")

df_old = convert_to_original_format(movies, credits)
df_new.head()
df_old.head()
df = df_new.copy()

df['vote_classes'] = pd.cut(df['vote_average'],4, labels=["low", "medium-low","medium-high","high"])
df['log_budget'] = np.log(df['budget'])

df['log_popularity'] = np.log(df['popularity'])

df['log_vote_average'] = np.log(df['vote_average'])

df['log_vote_count'] = np.log(df['vote_count'])

df['log_revenue']= np.log(df['revenue'])

df['log_runtime']= np.log(df['runtime'])

df_copy = df

df=df[df.columns[-6:]]

df_copy2 = df



df=df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

df=df.dropna(axis=1)
df_new.head()
scatter_matrix(df,alpha=0.2, figsize=(20, 20), diagonal='kde')
df = df_old
df['country'].unique()
df_countries = df['title_year'].groupby(df['country']).count()

df_countries = df_countries.reset_index()

df_countries.rename(columns ={'title_year':'count'}, inplace = True)

df_countries = df_countries.sort_values('count', ascending = False)

df_countries.reset_index(drop=True, inplace = True)
sns.set_context("poster", font_scale=0.6)

plt.rc('font', weight='bold')

f, ax = plt.subplots(figsize=(11, 6))

labels = [s[0] if s[1] > 80 else ' ' 

          for index, s in  df_countries[['country', 'count']].iterrows()]

sizes  = df_countries['count'].values

explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(df_countries))]

ax.pie(sizes, explode = explode, labels = labels,

       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',

       shadow=False, startangle=45)

ax.axis('equal')

ax.set_title('% of films per country',

             bbox={'facecolor':'k', 'pad':5},color='w', fontsize=16);
data = dict(type='choropleth',

locations = df_countries['country'],

locationmode = 'country names', z = df_countries['count'],

text = df_countries['country'], colorbar = {'title':'Films nb.'},

colorscale=[[0, 'rgb(224,255,255)'],

            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],

            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],

            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],

            [1, 'rgb(227,26,28)']],    

reversescale = False)



layout = dict(title='Number of films in the TMDB database',

geo = dict(showframe = True, projection={'type':'Mercator'}))



choromap = go.Figure(data = [data], layout = layout)

iplot(choromap, validate=False)
df = df_new.copy()
liste_genres = set()

for s in df['genres'].str.split('|'):

    liste_genres = set().union(s, liste_genres)

liste_genres = list(liste_genres)

liste_genres.remove('')



liste_genres
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
mean_per_genre.sort_values('mean_votes_average', ascending=False).head()
mean_per_genre.sort_values('mean_budget', ascending=False).head()
mean_per_genre.sort_values('mean_revenue', ascending=False).head()
mean_per_genre.sort_values('profit', ascending=False).head()
from datetime import datetime



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
profit_genre = revenue_genre[revenue_genre.columns[0:29]]-budget_genre[budget_genre.columns[0:29]]

profit_genre
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(budget_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
temp = budget_genre

temp[2013]=temp[2013].replace(2.550000e+08, 0)



fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(temp.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(revenue_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
temp2 = revenue_genre

temp2[1994] = temp2[1994].replace(788241776.0, 0)

temp2[1992] = temp2[1992].replace(504050219.0, 0)



fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(temp2.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(profit_genre, xticklabels=3, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(vote_avg_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
from datetime import datetime



df_genre = pd.DataFrame(columns = ['genre', 'cgenres', 'budget', 'gross', 'year'])

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
df = df_new.copy()
liste_keywords = set()

for s in df['keywords'].str.split('|'):

    liste_keywords = set().union(s, liste_keywords)

liste_keywords = list(liste_keywords)

liste_keywords.remove('')
def count_word(df, ref_col, liste):

    keyword_count = dict()

    for s in liste: keyword_count[s] = 0

    for liste_keywords in df[ref_col].str.split('|'):        

        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        

        for s in [s for s in liste_keywords if s in liste]: 

            if pd.notnull(s): keyword_count[s] += 1

    #______________________________________________________________________

    # convert the dictionary in a list to sort the keywords by frequency

    keyword_occurences = []

    for k,v in keyword_count.items():

        keyword_occurences.append([k,v])

    keyword_occurences.sort(key = lambda x:x[1], reverse = True)

    return keyword_occurences, keyword_count



keyword_occurences, dum = count_word(df, 'keywords', liste_keywords)

keyword_occurences[:5]
#We collect all the keywords:

def keywords_inventory(dataframe, colonne = 'keywords'):

    PS = nltk.stem.PorterStemmer()

    keywords_roots  = dict()  # collect the words / root

    keywords_select = dict()  # association: root <-> keyword

    category_keys = []

    icount = 0

    for s in dataframe[colonne]:

        if pd.isnull(s): continue

        for t in s.split('|'):

            t = t.lower() ; racine = PS.stem(t)

            if racine in keywords_roots:                

                keywords_roots[racine].add(t)

            else:

                keywords_roots[racine] = {t}

    

    for s in keywords_roots.keys():

        if len(keywords_roots[s]) > 1:  

            min_length = 1000

            for k in keywords_roots[s]:

                if len(k) < min_length:

                    clef = k ; min_length = len(k)            

            category_keys.append(clef)

            keywords_select[s] = clef

        else:

            category_keys.append(list(keywords_roots[s])[0])

            keywords_select[s] = list(keywords_roots[s])[0]

                   

    print("Nb of keywords in variable '{}': {}".format(colonne,len(category_keys)))

    return category_keys, keywords_roots, keywords_select



keywords, keywords_roots, keywords_select = keywords_inventory(df, colonne = 'keywords')
# Plot of a sample of keywords that appear in close varieties 

#------------------------------------------------------------

icount = 0

for s in keywords_roots.keys():

    if len(keywords_roots[s]) > 1: 

        icount += 1

        if icount < 15: print(icount, keywords_roots[s], len(keywords_roots[s]))
def remplacement_df_keywords(df, dico_remplacement, roots = False):

    df_new = df.copy(deep = True)

    for index, row in df_new.iterrows():

        chaine = row['keywords']

        if pd.isnull(chaine): continue

        nouvelle_liste = []

        for s in chaine.split('|'): 

            clef = PS.stem(s) if roots else s

            if clef in dico_remplacement.keys():

                nouvelle_liste.append(dico_remplacement[clef])

            else:

                nouvelle_liste.append(s)       

        df_new.set_value(index, 'keywords', '|'.join(nouvelle_liste)) 

    return df_new



df_keywords_cleaned = remplacement_df_keywords(df, keywords_select,roots = True)

df_keywords_cleaned.head()
def get_synonymes(word):

    lemma = set()

    for ss in wordnet.synsets(word):

        for w in ss.lemma_names():

            #_______________________________

            # We just get the 'nouns':

            index = ss.name().find('.')+1

            if ss.name()[index] == 'n': lemma.add(w.lower().replace('_',' '))

    return lemma



def test_keyword(mot, key_count, threshold):

    return (False , True)[key_count.get(mot, 0) >= threshold]



keyword_occurences.sort(key = lambda x:x[1], reverse = False)

key_count = dict()

for s in keyword_occurences:

    key_count[s[0]] = s[1]

#__________________________________________________________________________

# Creation of a dictionary to replace keywords by higher frequency keywords

remplacement_mot = dict()

icount = 0

for index, [mot, nb_apparitions] in enumerate(keyword_occurences):

    if nb_apparitions > 5: continue  # only the keywords that appear less than 5 times

    lemma = get_synonymes(mot)

    if len(lemma) == 0: continue     # case of the plurals

    #_________________________________________________________________

    liste_mots = [(s, key_count[s]) for s in lemma 

                  if test_keyword(s, key_count, key_count[mot])]

    liste_mots.sort(key = lambda x:(x[1],x[0]), reverse = True)    

    if len(liste_mots) <= 1: continue       # no replacement

    if mot == liste_mots[0][0]: continue    # replacement by himself

    icount += 1

    if  icount < 8:

        print('{:<12} -> {:<12} (init: {})'.format(mot, liste_mots[0][0], liste_mots))    

    remplacement_mot[mot] = liste_mots[0][0]



print(90*'_'+'\n'+'The replacement concerns {}% of the keywords.'

      .format(round(len(remplacement_mot)/len(keywords)*100,2)))



# 2 successive replacements

#---------------------------

print('Keywords that appear both in keys and values:'.upper()+'\n'+45*'-')

icount = 0

for s in remplacement_mot.values():

    if s in remplacement_mot.keys():

        icount += 1

        if icount < 10: print('{:<20} -> {:<20}'.format(s, remplacement_mot[s]))



for key, value in remplacement_mot.items():

    if value in remplacement_mot.keys():

        remplacement_mot[key] = remplacement_mot[value]
# replacement of keyword varieties by the main keyword

#----------------------------------------------------------

df_keywords_synonyms = remplacement_df_keywords(df_keywords_cleaned, remplacement_mot, roots = False)   

keywords, keywords_roots, keywords_select = keywords_inventory(df_keywords_synonyms, colonne = 'keywords')



# New count of keyword occurences

#-------------------------------------

keywords.remove('')

new_keyword_occurences, keywords_count = count_word(df_keywords_synonyms,

                                                    'keywords',keywords)
# deletion of keywords with low frequencies

#-------------------------------------------

def remplacement_df_low_frequency_keywords(df, keyword_occurences):

    df_new = df.copy(deep = True)

    key_count = dict()

    for s in keyword_occurences: 

        key_count[s[0]] = s[1]    

    for index, row in df_new.iterrows():

        chaine = row['keywords']

        if pd.isnull(chaine): continue

        nouvelle_liste = []

        for s in chaine.split('|'): 

            if key_count.get(s, 4) > 3: nouvelle_liste.append(s)

        df_new.set_value(index, 'keywords', '|'.join(nouvelle_liste))

    return df_new



# Creation of a dataframe where keywords of low frequencies are suppressed

#-------------------------------------------------------------------------

df_keywords_occurence = remplacement_df_low_frequency_keywords(df_keywords_synonyms, new_keyword_occurences)

keywords, keywords_roots, keywords_select = keywords_inventory(df_keywords_occurence, colonne = 'keywords')   
df_keywords= df_keywords_occurence

keyword_list = set()

for s in df_keywords['keywords'].str.split('|'):

    keyword_list = set().union(s, keyword_list)

keyword_list = list(keyword_list)

keyword_list.remove('')



df_reduced = df_keywords[['title','vote_average','release_date','runtime','budget','revenue']].reset_index(drop=True)



for keyword in keyword_list:

    df_reduced[keyword] = df['keywords'].str.contains(keyword).apply(lambda x:1 if x else 0)

df_reduced[:5]



df_reduced.head()
mean_per_keyword = pd.DataFrame(keyword_list)



#Mean votes average

newArray1 = []*len(keyword_list)

for keyword in keyword_list:

    newArray1.append(df_reduced.groupby(keyword, as_index=True)['vote_average'].mean())

    

#Mean budget

newArray2 = []*len(keyword_list)

for keyword in keyword_list:

    newArray2.append(df_reduced.groupby(keyword, as_index=True)['budget'].mean())

    

#Mean revenue

newArray3 = []*len(keyword_list)

for keyword in keyword_list:

    newArray3.append(df_reduced.groupby(keyword, as_index=True)['revenue'].mean())



mean_per_keyword['mean_vote_average']=list(pd.DataFrame(newArray1)[1])

mean_per_keyword['mean_budget']=list(pd.DataFrame(newArray2)[1])

mean_per_keyword['mean_revenue']=list(pd.DataFrame(newArray3)[1])

mean_per_keyword['profit']= (mean_per_keyword['mean_revenue'] - mean_per_keyword['mean_budget'])
mean_per_keyword.sort_values('mean_vote_average', ascending=False).head()
mean_per_keyword.sort_values('mean_budget', ascending=False).head()
mean_per_keyword.sort_values('mean_revenue', ascending=False).head()
mean_per_keyword.sort_values('profit', ascending=False).head()
fig = plt.figure(1, figsize=(18,13))

trunc_occurences = new_keyword_occurences[0:50]

# LOWER PANEL: HISTOGRAMS

ax2 = fig.add_subplot(2,1,2)

y_axis = [i[1] for i in trunc_occurences]

x_axis = [k for k,i in enumerate(trunc_occurences)]

x_label = [i[0] for i in trunc_occurences]

plt.xticks(rotation=85, fontsize = 15)

plt.yticks(fontsize = 15)

plt.xticks(x_axis, x_label)

plt.ylabel("Nb. of occurences", fontsize = 18, labelpad = 10)

ax2.bar(x_axis, y_axis, align = 'center', color='g')

#_______________________

plt.title("Keywords popularity",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)

plt.show()
Df1 = pd.DataFrame(trunc_occurences)

Df2 = mean_per_keyword

result = Df1.merge(Df2, left_on=0, right_on=0, how='inner')



result = result.rename(columns ={0:'keyword', 1:'occurences'})



result.sort_values('mean_vote_average', ascending= False)
import matplotlib.pyplot as plt



ax = result.plot.bar(x = 'keyword', y='mean_vote_average', title="mean vote average",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label = "mean vote average")

ax.set_ylim(5, 8)

ax.axhline(y=result['mean_vote_average'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()



import matplotlib.pyplot as plt



ax = result.plot.bar(x = 'keyword', y='mean_budget', title="mean budget",

                     figsize=(15,4), legend=True, fontsize=12, color='blue', label="mean budget")

ax.axhline(y=result['mean_budget'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()
ax = result.plot.bar(x = 'keyword', y='mean_revenue', title="mean revenue",

                     figsize=(15,4), legend=True, fontsize=12, color='yellow', label="mean revenue")

ax.axhline(y=result['mean_revenue'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()
result['profit'] = result['mean_revenue'] - result['mean_budget']



ax = result.plot.bar(x = 'keyword', y='profit', title="profit",

                     figsize=(15,4), legend=True, fontsize=12, color='red', label="profit")

ax.axhline(y=result['profit'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()
df = df_old
columns = ['homepage', 'plot_keywords', 'language', 'overview', 'popularity', 'tagline',

           'original_title', 'num_voted_users', 'country', 'spoken_languages', 'duration',

          'production_companies', 'production_countries', 'status']



df = df.drop(columns, axis=1)
liste_genres = set()

for s in df['genres'].str.split('|'):

    liste_genres = set().union(s, liste_genres)

liste_genres = list(liste_genres)

liste_genres.remove('')



df_reduced = df[['actor_1_name', 'vote_average',

                 'title_year', 'movie_title', 'gross', 'budget']].reset_index(drop = True)

for genre in liste_genres:

    df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)



df_reduced2 = df[['actor_2_name', 'vote_average',

                 'title_year', 'movie_title', 'gross', 'budget']].reset_index(drop = True)

for genre in liste_genres:

    df_reduced2[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)



df_reduced3 = df[['actor_3_name', 'vote_average',

                 'title_year', 'movie_title', 'gross', 'budget']].reset_index(drop = True)

for genre in liste_genres:

    df_reduced3[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)
df_reduced = df_reduced.rename(columns={'actor_1_name': 'actor'})

df_reduced2 = df_reduced2.rename(columns={'actor_2_name': 'actor'})

df_reduced3 = df_reduced3.rename(columns={'actor_3_name': 'actor'})



total = [df_reduced, df_reduced2, df_reduced3]

df_total = pd.concat(total)

df_total.head()
df_actors = df_total.groupby('actor').mean()

df_actors.loc[:, 'favored_genre'] = df_actors[liste_genres].idxmax(axis = 1)

df_actors.drop(liste_genres, axis = 1, inplace = True)

df_actors = df_actors.reset_index()
df_appearance = df_total[['actor', 'title_year']].groupby('actor').count()

df_appearance = df_appearance.reset_index(drop = True)

selection = df_appearance['title_year'] > 9

selection = selection.reset_index(drop = True)

most_prolific = df_actors[selection]
most_prolific.sort_values('vote_average', ascending=False).head()
most_prolific.sort_values('gross', ascending=False).head()
most_prolific.sort_values('budget', ascending=False).head()
genre_count = []

for genre in liste_genres:

    genre_count.append([genre, df_reduced[genre].values.sum()])

genre_count.sort(key = lambda x:x[1], reverse = True)

labels, sizes = zip(*genre_count)

labels_selected = [n if v > sum(sizes) * 0.01 else '' for n, v in genre_count]

reduced_genre_list = labels[:19]

trace=[]

for genre in reduced_genre_list:

    trace.append({'type':'scatter',

                  'mode':'markers',

                  'y':most_prolific.loc[most_prolific['favored_genre']==genre,'gross'],

                  'x':most_prolific.loc[most_prolific['favored_genre']==genre,'budget'],

                  'name':genre,

                  'text': most_prolific.loc[most_prolific['favored_genre']==genre,'actor'],

                  'marker':{'size':10,'opacity':0.7,

                            'line':{'width':1.25,'color':'black'}}})

layout={'title':'Actors favored genres',

       'xaxis':{'title':'mean budget'},

       'yaxis':{'title':'mean revenue'}}

fig=Figure(data=trace,layout=layout)

pyo.iplot(fig)
selection = df_appearance['title_year'] > 20

most_prolific = df_actors[selection]

most_prolific
class Trace():

    #____________________

    def __init__(self, color):

        self.mode = 'markers'

        self.name = 'default'

        self.title = 'default title'

        self.marker = dict(color=color, size=110,

                           line=dict(color='white'), opacity=0.7)

        self.r = []

        self.t = []

    #______________________________

    def set_color(self, color):

        self.marker = dict(color = color, size=110,

                           line=dict(color='white'), opacity=0.7)

    #____________________________

    def set_name(self, name):

        self.name = name

    #____________________________

    def set_title(self, title):

        self.na = title

    #___________________________

    def set_actor(self, actor):

        self.actor = actor

    

    #__________________________

    def set_values(self, r, t):

        self.r = np.array(r)

        self.t = np.array(t)
names =['Morgan Freeman']

df2 = df_reduced[df_reduced['actor'] == 'Morgan Freeman']

total_count  = 0

years = []

imdb_score = []

genre = []

titles = []

actor = []

for s in liste_genres:

    icount = df2[s].sum()

    #__________________________________________________________________

    # Here, we set the limit to 3 because of a bug in plotly's package

    if icount > 3: 

        total_count += 1

        genre.append(s)

        actor.append(list(df2[df2[s] ==1 ]['actor']))

        years.append(list(df2[df2[s] == 1]['title_year']))

        imdb_score.append(list(df2[df2[s] == 1]['vote_average'])) 

        titles.append(list(df2[df2[s] == 1]['movie_title']))

max_y = max([max(s) for s in years])

min_y = min([min(s) for s in years])

year_range = max_y - min_y



years_normed = []

for i in range(total_count):

    years_normed.append( [360/total_count*((an-min_y)/year_range+i) for an in years[i]])

    

color = ('royalblue', 'grey', 'wheat', 'c', 'firebrick', 'seagreen', 'lightskyblue',

          'lightcoral', 'yellowgreen', 'gold', 'tomato', 'violet', 'aquamarine', 'chartreuse', 'red')



trace = [Trace(color[i]) for i in range(total_count)]

tr    = []

for i in range(total_count):

    trace[i].set_name(genre[i])

    trace[i].set_title(titles[i])

    trace[i].set_values(np.array(imdb_score[i]),

                        np.array(years_normed[i]))

    tr.append(go.Scatter(r      = trace[i].r,

                         t      = trace[i].t,

                         mode   = trace[i].mode,

                         name   = trace[i].name,

                         marker = trace[i].marker,

#                         text   = ['default title' for j in range(len(trace[i].r))], 

                         hoverinfo = 'all'

                        ))        

layout = go.Layout(

    title='Morgan Freeman movies',

    font=dict(

        size=15

    ),

    plot_bgcolor='rgb(223, 223, 223)',

    angularaxis=dict(        

        tickcolor='rgb(253,253,253)'

    ),

    hovermode='Closest',

)

fig = go.Figure(data = tr, layout=layout)

pyo.iplot(fig)
df = df_old
def create_comparison_database(name, value, x, no_films):

    

    comparison_df = df.groupby(name, as_index=False)

    

    if x == 'mean':

        comparison_df = comparison_df.mean()

    elif x == 'median':

        comparison_df = comparison_df.median()

    elif x == 'sum':

        comparison_df = comparison_df.sum() 

    

    # Create database with either name of directors or actors, the value being compared i.e. 'revenue',

    # and number of films they're listed with. Then sort by value being compared.

    name_count_key = df[name].value_counts().to_dict()

    comparison_df['films'] = comparison_df[name].map(name_count_key)

    comparison_df.sort_values(value, ascending=False, inplace=True)

    comparison_df[name] = comparison_df[name].map(str) + " (" + comparison_df['films'].astype(str) + ")"

   # create a Series with the name as the index so it can be plotted to a subgrid

    comp_series = comparison_df[comparison_df['films'] >= no_films][[name, value]][10::-1].set_index(name).ix[:,0]

    

    return comp_series
fig = plt.figure(figsize=(18,6))



# Director_name

plt.subplot2grid((2,3),(0,0), rowspan = 2)

create_comparison_database('director_name','gross','sum', 4).plot(kind='barh', color='#006600')

plt.legend().set_visible(False)

plt.title("Total revenue for Directors with 4+ Films")

plt.ylabel("Director (no. films)")

plt.xlabel("Revenue")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','gross','mean', 4).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Average revenue for Directors with 4+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("Revenue")



plt.tight_layout()
fig = plt.figure(figsize=(18,6))



# Director_name

plt.subplot2grid((2,3),(0,0), rowspan = 2)

create_comparison_database('director_name','budget','mean', 4).plot(kind='barh', color='#006600')

plt.legend().set_visible(False)

plt.title("Average budget for Directors with 4+ Filmss")

plt.ylabel("Director (no. films)")

plt.xlabel("Budget")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','vote_average','mean', 4).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Mean IMDB Score for Directors with 4+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("IMDB Score")

plt.xlim(0,10)



plt.tight_layout()
fig = plt.figure(figsize=(18,6))



# Director_name

plt.subplot2grid((2,3),(0,0), rowspan = 2)

create_comparison_database('director_name','budget','mean', 10).plot(kind='barh', color='#006600')

plt.legend().set_visible(False)

plt.title("Average budget for Directors with 15+ Filmss")

plt.ylabel("Director (no. films)")

plt.xlabel("Budget")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','vote_average','mean', 10).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Mean IMDB Score for Directors with 15+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("IMDB Score")

plt.xlim(0,10)



plt.tight_layout()
df = df_new.copy()



df['vote_classes'] = pd.cut(df['vote_average'],100, labels=range(100))



my_imputer = Imputer()



X2 = my_imputer.fit_transform(df[['runtime']])

df['runtime'] = X2



col = ['budget', 'runtime']

X = df[col]

y = df['vote_classes']
# instantiate the model (with the default parameters)

knn_score = KNeighborsClassifier()



# fit the model with data (occurs in-place)

knn_score.fit(X, y)
y = df['revenue']



# instantiate the model (with the default parameters)

knn_rev = KNeighborsClassifier()



# fit the model with data (occurs in-place)

knn_rev.fit(X, y)
df = df_new.copy()



my_imputer = Imputer()



X2 = my_imputer.fit_transform(df[['runtime']])

df['runtime'] = X2



budget = ['budget','runtime']

training = df[budget]



target= ['vote_average']

target = df[target]



X = training.values

y = target.values



X_train1, X_test1, y_train1, y_test1 = train_test_split(

    X, y, test_size=0.3)



# Create linear regression object

regr1 = linear_model.LinearRegression()



# Train the model using the training sets

regr1.fit(X_train1, y_train1)



# Make predictions using the testing set

y_pred_lr1 = regr1.predict(X_test1)



print(regr1.coef_, regr1.intercept_)

print(r2_score(y_test1, y_pred_lr1))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test1[:,1], y_test1, label="Real score");

plt.scatter(X_test1[:,1], y_pred_lr1, c='r',label="Predicted score");

plt.xlabel("Budget");

plt.ylabel('Score');

plt.legend(loc=2);
target = ['revenue']

target = df[target]



X = training.values

y = target.values



X_train2, X_test2, y_train2, y_test2 = train_test_split(

    X, y, test_size=0.3)



# Create linear regression object

regr2 = linear_model.LinearRegression()



# Train the model using the training sets

regr2.fit(X_train2, y_train2)



# Make predictions using the testing set

y_pred_lr2 = regr2.predict(X_test2)



print(regr2.coef_, regr2.intercept_)

print(r2_score(y_test2, y_pred_lr2))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test2[:,1], y_test2, label="Real revenue");

plt.scatter(X_test2[:,1], y_pred_lr2, c='r',label="Predicted revenue");

plt.ylabel('Revenue');

plt.legend(loc=2);
# Create linear regression object

rf1 = RandomForestRegressor(1)



# Train the model using the training sets

rf1.fit(X_train1, y_train1)



# Make predictions using the testing set

y_pred_rf1 = rf1.predict(X_test1)
f = plt.figure(figsize=(10,5))

plt.scatter(X_test1[:,1], y_test1, s=50,label="Real Score");

plt.scatter(X_test1[:,1], y_pred_rf1,s=100, c='r',label="Predicted Score");

plt.ylabel("Score");

plt.legend(loc=2);
# Create linear regression object

rf2 = RandomForestRegressor(1)



# Train the model using the training sets

rf2.fit(X_train2, y_train2)



# Make predictions using the testing set

y_pred_rf2 = rf2.predict(X_test2)
f = plt.figure(figsize=(10,5))

plt.scatter(X_test2[:,1], y_test2, s=50,label="Real Score");

plt.scatter(X_test2[:,1], y_pred_rf2,s=100, c='r',label="Predicted Score");

plt.ylabel("Score");

plt.legend(loc=2);
error_lr = mean_squared_error(y_test1,y_pred_lr1)

error_rf = mean_squared_error(y_test1,y_pred_rf1)

print(error_lr)

print(error_rf)



f = plt.figure(figsize=(10,5))

plt.bar(range(2),[error_lr,error_rf])

plt.xlabel("Classifiers");

plt.ylabel("Mean Squared Error of the Score");

plt.xticks(range(2),['Linear Regression','Random Forest'])

plt.legend(loc=2);
error_lr = mean_squared_error(y_test2,y_pred_lr2)

error_rf = mean_squared_error(y_test2,y_pred_rf2)

print(error_lr)

print(error_rf)



f = plt.figure(figsize=(10,5))

plt.bar(range(2),[error_lr,error_rf])

plt.xlabel("Classifiers");

plt.ylabel("Mean Squared Error of the Revenue");

plt.xticks(range(2),['Linear Regression','Random Forest'])

plt.legend(loc=2);
print("Score predictions:")



pred = knn_score.predict([[250000000, 165]])

print(pred)



pred = regr1.predict([[250000000, 165]])

print(pred)



pred = rf1.predict([[250000000, 165]])

print(pred)



print("\nRevenue predictions:")



pred = knn_rev.predict([[250000000, 165]])

print(pred)



pred = regr2.predict([[250000000, 165]])

print(pred)



pred = rf2.predict([[250000000, 165]])

print(pred)
df = df_new.copy()



t = df['release_date']

t = pd.to_datetime(t)

t = t.dt.year

df['release_year'] = t



my_imputer = Imputer()



X2 = my_imputer.fit_transform(df[['runtime']])

df['runtime'] = X2



X2 = my_imputer.fit_transform(df[['vote_count']])

df['vote_count'] = X2



X2 = my_imputer.fit_transform(df[['release_year']])

df['release_year'] = X2



X2 = my_imputer.fit_transform(df[['popularity']])

df['popularity'] = X2



predictors = ['budget', 'vote_count', 'vote_average','runtime', 'release_year', 'popularity']

training = df[predictors]



target= ['revenue']

target = df[target]



X = training.values

y = target.values



X_train1, X_test1, y_train1, y_test1 = train_test_split(

    X, y, test_size=0.3)



# Create linear regression object

regr1 = linear_model.LinearRegression()



# Train the model using the training sets

regr1.fit(X_train1, y_train1)



# Make predictions using the testing set

y_pred_lr1 = regr1.predict(X_test1)



print(r2_score(y_test1, y_pred_lr1))



f = plt.figure(figsize=(10,5))

plt.scatter(X_test1[:,1], y_test1, label="Real Revenue");

plt.scatter(X_test1[:,1], y_pred_lr1, c='r',label="Predicted Revenue");

plt.ylabel('Revenue');

plt.legend(loc=2);