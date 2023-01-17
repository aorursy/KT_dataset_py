

from sklearn.preprocessing import Imputer

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 

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



movies = pd.read_csv('../input/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb_5000_credits.csv')



movies.head()
credits.head()
(credits['title']==movies['title']).describe()
del credits['title']

del credits['movie_id']

movie_df = pd.concat([movies, credits], axis=1)

movie_df.head()
newCols = ['title','release_date','popularity','vote_average','vote_count',

           'budget','revenue','genres','keywords','cast','crew','tagline', 'runtime', 'production_companies', 

           'production_countries', 'status']



df2 = movie_df[newCols]

df2.head()
df2.describe().round()
my_imputer = Imputer()



temp=df2

X2 = my_imputer.fit_transform(df2[['runtime']])

df2['runtime'] = X2

df2.describe().round()
#df2['vote_classes'] = pd.cut(df2['vote_average'],10, labels=["1", "2","3","4","5","6","7","8","9","10"])

df2['vote_classes'] = pd.cut(df2['vote_average'],4, labels=["low", "medium-low","medium-high","high"])
df2['log_budget'] = np.log(df2['budget'])

df2['log_popularity'] = np.log(df2['popularity'])

df2['log_vote_average'] = np.log(df2['vote_average'])

df2['log_vote_count'] = np.log(df2['vote_count'])

df2['log_revenue']= np.log(df2['revenue'])

df2['log_runtime']= np.log(df2['runtime'])

df3=df2[df2.columns[-5:]]



#df3.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

df3=df3[df3.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

df3=df3.dropna(axis=1)

#df3[~df3.isin([np.nan, np.inf, -np.inf]).any(1)]

from pandas.plotting import scatter_matrix

scatter_matrix(df3,alpha=0.2, figsize=(20, 20), diagonal='kde')
Early_df = df2[df2.columns[0:16]]

Early_df.head()
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



#vote_avg_genre.index = vote_avg_genre['genre']
budget_genre.index = budget_genre['genre']

budget_genre
revenue_genre.index = revenue_genre['genre']

revenue_genre

vote_avg_genre.index = vote_avg_genre['genre']

vote_avg_genre
#revenue_genre[revenue_genre.columns[1]]

#budget_genre[budget_genre.columns[1]]

profit_genre = revenue_genre[revenue_genre.columns[0:29]]-budget_genre[budget_genre.columns[0:29]]

#df2[df2.columns[0:16]
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(budget_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)


fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(revenue_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(profit_genre, xticklabels=3, cmap=cmap, linewidths=0.05)
fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(vote_avg_genre.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
temp = budget_genre

temp[2013]=temp[2013].replace(2.550000e+08, 0)



fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(temp.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
temp2 = revenue_genre

temp2[1994] = temp2[1994].replace(788241776.0, 0)

temp2[1992] = temp2[1992].replace(504050219.0, 0)



fig, ax = plt.subplots(figsize=(9,9))

cmap = sns.cubehelix_palette(start = 1.5, rot = 1.5, as_cmap = True)

sns.heatmap(temp2.ix[:,0:30], xticklabels=3, cmap=cmap, linewidths=0.05)
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
credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")

movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")



del credits['title']

df = pd.concat([movies, credits], axis=1)



df['keywords'] = df['keywords'].apply(pipe_flatten_names)



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

new_keyword_occurences[:5]
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
df_keywords_occurence.head()
df_keywords= df_keywords_occurence

keyword_list = set()

for s in df_keywords['keywords'].str.split('|'):

    keyword_list = set().union(s, keyword_list)

keyword_list = list(keyword_list)

keyword_list.remove('')

keyword_list[:5]
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



mean_per_keyword.sort_values('mean_vote_average', ascending=False).head()
mean_per_keyword.sort_values('mean_budget', ascending=False).head()
mean_per_keyword.sort_values('mean_revenue', ascending=False).head()
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
result['mean_vote_average'].mean()
import matplotlib.pyplot as plt



ax = result.plot.bar(x = 'keyword', y='mean_vote_average', title="mean vote average",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label = "mean vote average")

ax.set_ylim(5, 8)

ax.axhline(y=result['mean_vote_average'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()

import matplotlib.pyplot as plt



ax = result.plot.bar(x = 'keyword', y='mean_budget', title="mean budget",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label="mean budget")

ax.axhline(y=result['mean_budget'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()

result.sort_values('mean_budget').head()
ax = result.plot.bar(x = 'keyword', y='mean_revenue', title="mean revenue",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label="mean revenue")

ax.axhline(y=result['mean_revenue'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()
result['profit'] = result['mean_revenue'] - result['mean_budget']

result.head()
ax = result.plot.bar(x = 'keyword', y='profit', title="profit",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label="profit")

ax.axhline(y=result['profit'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()
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

df = convert_to_original_format(movies, credits)
df.head()
df3 = df # We store a copy of the dataframe for later use
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

       'xaxis':{'title':'mean year of activity'},

       'yaxis':{'title':'mean score'}}

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
df = df3
def create_comparison_database(name, value, x, no_films):

    

    comparison_df = df3.groupby(name, as_index=False)

    

    if x == 'mean':

        comparison_df = comparison_df.mean()

    elif x == 'median':

        comparison_df = comparison_df.median()

    elif x == 'sum':

        comparison_df = comparison_df.sum() 

    

    # Create database with either name of directors or actors, the value being compared i.e. 'gross',

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

plt.title("Total Gross for Directors with 4+ Films")

plt.ylabel("Director (no. films)")

plt.xlabel("Gross (in billons)")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','gross','mean', 4).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Average revenue for Directors with 4+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("Gross (in billons)")



plt.tight_layout()
fig = plt.figure(figsize=(18,6))



# Director_name

plt.subplot2grid((2,3),(0,0), rowspan = 2)

create_comparison_database('director_name','budget','mean', 4).plot(kind='barh', color='#006600')

plt.legend().set_visible(False)

plt.title("Average budget for Directors with 4+ Filmss")

plt.ylabel("Director (no. films)")

plt.xlabel("Budget (in billons)")



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

plt.xlabel("Budget (in billons)")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','vote_average','mean', 10).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Mean IMDB Score for Directors with 15+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("IMDB Score")

plt.xlim(0,10)



plt.tight_layout()
df2.head()
df2 = Early_df
df2['log_budget'] = np.log(df2['budget'])

df2['log_popularity'] = np.log(df2['popularity'])

df2['log_vote_average'] = np.log(df2['vote_average'])

df2['log_vote_count'] = np.log(df2['vote_count'])

df2['log_revenue']= np.log(df2['revenue'])

df2['log_runtime']= np.log(df2['runtime'])

df3=df2[df2.columns[-6:]]



df3=df3[df3.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

df3=df3.dropna(axis=1)

df3.head()
num_list = ['budget','popularity','revenue','runtime','vote_average','vote_count']

movie_num = df2[num_list]

movie_num.head()
df3.head()
movie_num['revenue'] .plot(kind='hist')
df3['log_revenue'].plot(kind='hist')
f, ax = plt.subplots(figsize=(12,10))

plt.title('Pearson Correlation of Movie Features')

sns.heatmap(movie_num.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True,

           cmap="YlGnBu", linecolor='black', annot=True)
num_list = ['budget','runtime','vote_average']

movie_num = df2[num_list]

movie_num.head()
training_list = ['budget','runtime']

training = movie_num[training_list]

target = movie_num['vote_average']
X = training.values

y = target.values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)
from sklearn import linear_model

# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)
f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,1], y_test, s=50,label="Real vote_average");

plt.scatter(X_test[:,1], y_pred_lr,s=100, c='r',label="Predicted vote_average");

plt.ylabel("vote_average");

plt.legend(loc=2);
from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)
f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,1], y_test, s=50,label="Real vote_average");

plt.scatter(X_test[:,1], y_pred_rf,s=100, c='r',label="Predited vote_average");

plt.ylabel("vote_average");

plt.legend(loc=2);
from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



print(error_lr)

print(error_rf)
f = plt.figure(figsize=(10,5))

plt.bar(range(2),[error_lr,error_rf])

plt.xlabel("Classifiers");

plt.ylabel("Mean Squared Error of the vote_average");

plt.xticks(range(2),['Linear Regression','Random Forest'])

plt.legend(loc=2);
num_list = ['budget','revenue', "runtime"]

movie_num = df2[num_list]

 

training_list = ['budget','runtime']

training = movie_num[training_list]

target = movie_num['revenue']



X = training.values

y = target.values

 

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)



from sklearn import linear_model

# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)

 

f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,1], y_test, s=50,label="Real vote_average");

plt.scatter(X_test[:,1], y_pred_lr,s=100, c='r',label="Predicted vote_average");

plt.ylabel("vote_average");

plt.legend(loc=2);

 

 #-----------------------------------------

from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)

 

f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,1], y_test, s=50,label="Real vote_average");

plt.scatter(X_test[:,1], y_pred_rf,s=100, c='r',label="Predited vote_average");

plt.ylabel("vote_average");

plt.legend(loc=2);

 

from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



print(error_lr)

print(error_rf)

 

f = plt.figure(figsize=(10,5))

plt.bar(range(2),[error_lr,error_rf])

plt.xlabel("Classifiers");

plt.ylabel("Mean Squared Error of the vote_average");

plt.xticks(range(2),['Linear Regression','Random Forest'])

plt.legend(loc=2);
num_list = ['budget','popularity','revenue','runtime','vote_average','vote_count']

movie_num = df2[num_list]

movie_num.head()
df2['vote_classes'] = pd.cut(df2['vote_average'],4, labels=["low",'medium-low','medium-high','high'])

num_list = ['budget','popularity','revenue','runtime','vote_count','vote_classes']

movie_num = df2[num_list]

movie_num.head()



from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=10)

y, _ = pd.factorize(train['vote_classes'])

clf.fit(train[features],y)



preds = df.vote_classes[clf.predict(test[features])]

pd.crosstab(test['vote_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df2['vote_classes'] = pd.cut(df2['vote_average'],10, labels=range(10))
df3['vote_classes'] = pd.cut(df2['vote_average'],4, labels=['low','medium-low','medium-high','high'])

num_list = ['log_budget','log_popularity','log_revenue','log_runtime','log_vote_count','vote_classes']

movie_num = df3[num_list]

movie_num.head()



from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=1000)

y, _ = pd.factorize(train['vote_classes'])

clf.fit(train[features],y)



preds = df.vote_classes[clf.predict(test[features])]

pd.crosstab(test['vote_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df2['vote_classes'] = pd.cut(df2['vote_average'],4, labels=['low','medium-low','medium-high','high'])

num_list = ['budget','popularity','revenue','runtime','vote_count','vote_classes']

movie_num = df2[num_list]

movie_num.head()



from sklearn.preprocessing import StandardScaler



movie_num['normBudget'] = StandardScaler().fit_transform(movie_num['budget'].reshape(-1, 1))

movie_num['normPopularity'] = StandardScaler().fit_transform(movie_num['popularity'].reshape(-1, 1))

movie_num['normRevenue'] = StandardScaler().fit_transform(movie_num['revenue'].reshape(-1, 1))

movie_num['normVoteCount'] = StandardScaler().fit_transform(movie_num['vote_count'].reshape(-1, 1))

movie_num['normRuntime'] = StandardScaler().fit_transform(movie_num['runtime'].reshape(-1, 1))

#movie_num['vote_classes'] = pd.cut(movie_num['vote_average'],2, labels=[0,1])



movie_test = movie_num.drop(['budget','popularity','vote_count','revenue','runtime'],axis=1)

cols=['normBudget','normPopularity','normRevenue','normVoteCount','normRuntime','vote_classes']

movie_num = movie_test[cols]

#movie_test = movie_test[:-1] + movie_test[-1:]

movie_num.head()
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=1000)

y, _ = pd.factorize(train['vote_classes'])

clf.fit(train[features],y)



preds = df.vote_classes[clf.predict(test[features])]

pd.crosstab(test['vote_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df2['revenue_classes'] = pd.cut(df2['revenue'],3, labels=['low','medium','high'])

num_list = ['budget','popularity','vote_average','runtime','vote_count','revenue_classes']

movie_num = df2[num_list]

movie_num.head()



from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=100)

y, _ = pd.factorize(train['revenue_classes'])

clf.fit(train[features],y)



preds = df.revenue_classes[clf.predict(test[features])]

pd.crosstab(test['revenue_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df3['revenue_classes'] = pd.cut(df2['revenue'],3, labels=['low','medium','high'])

num_list = ['log_budget','log_popularity','log_vote_average','log_runtime','log_vote_count','revenue_classes']

movie_num = df3[num_list]

movie_num.head()



from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=1000)

y, _ = pd.factorize(train['revenue_classes'])

clf.fit(train[features],y)



preds = df.revenue_classes[clf.predict(test[features])]

pd.crosstab(test['revenue_classes'], preds.values, rownames=['actual'], colnames=['preds'])
df2['revenue_classes'] = pd.cut(df2['revenue'],3, labels=range(3))

num_list = ['budget','popularity','vote_average','runtime','vote_count','revenue_classes']

movie_num = df2[num_list]

movie_num.head()



from sklearn.preprocessing import StandardScaler



movie_num['normBudget'] = StandardScaler().fit_transform(movie_num['budget'].reshape(-1, 1))

movie_num['normPopularity'] = StandardScaler().fit_transform(movie_num['popularity'].reshape(-1, 1))

#movie_num['normVoteAverage'] = StandardScaler().fit_transform(movie_num['vote_average'].reshape(-1, 1))

movie_num['normRuntime'] = StandardScaler().fit_transform(movie_num['runtime'].reshape(-1, 1))

movie_num['normVoteCount'] = StandardScaler().fit_transform(movie_num['vote_count'].reshape(-1, 1))





#movie_num['revenue_classes'] = pd.cut(movie_num['vote_average'],2, labels=[0,1])



movie_test = movie_num.drop(['budget','popularity','runtime','vote_count',],axis=1)

cols=['normBudget','normPopularity','vote_average','normVoteCount','normRuntime','revenue_classes']

movie_num = movie_test[cols]

#movie_test = movie_test[:-1] + movie_test[-1:]

movie_num.head()
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np



df = movie_num

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



train, test = df[df['is_train']==True], df[df['is_train']==False]



features = df.columns[:5]

clf = RandomForestClassifier(n_jobs=1000)

y, _ = pd.factorize(train['revenue_classes'])

clf.fit(train[features],y)



preds = df.revenue_classes[clf.predict(test[features])]

pd.crosstab(test['revenue_classes'], preds.values, rownames=['actual'], colnames=['preds'])