import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
%matplotlib inline

from datetime import datetime
import datetime
import wordcloud as wc
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        saving=False


df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
df1=df1.rename({'movie_id': 'id'},axis=1)
df1.columns = ['id','title2','cast','crew']
df2= df2.merge(df1,on='id')
df2.info()
df2.drop(columns=['title2'],inplace=True)
df2.head(5)
df2.isnull()
sns.heatmap(df2.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df2.select_dtypes('object').nunique()
plt.figure(figsize=(25,6))


plt.subplot(2, 3, 1)
sns.distplot(df2['revenue'])

plt.subplot(2, 3, 2)
sns.distplot(df2['vote_count'])

plt.subplot(2, 3, 3)
sns.distplot(df2['budget'])

plt.subplot(2, 3, 4)
sns.distplot(df2['vote_average'].fillna(0).astype(int))

plt.subplot(2, 3, 5)
sns.distplot(df2['runtime'].fillna(0).astype(int))

plt.subplot(2, 3, 6)
sns.distplot(df2['popularity'].fillna(0).astype(int))

plt.suptitle('Checking for Skewness', fontsize = 15)
plt.show()
pop= df2.sort_values('revenue', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['revenue'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("revenue")
plt.title("revenue Movies")

pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")

movies = df2
movies['spoken_languages'] = movies['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


s = movies.apply(lambda x: pd.Series(x['spoken_languages']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'spoken_languages_count'
con_df = movies.drop('spoken_languages', axis=1).join(s)
con_df = pd.DataFrame(con_df['spoken_languages_count'].value_counts())
con_df['spoken_language'] = con_df.index
con_df.columns = ['num_spoken_language', 'spoken_language']


con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(100)
con_df = con_df[:5]

fig = plt.figure(figsize=(12,7))
sns.barplot(data = con_df, x='spoken_language', y = 'num_spoken_language')

plt.tight_layout()
movies = df2
movies['production_countries'] = movies['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = movies.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'countries'
con_df = movies.drop('production_countries', axis=1).join(s)
con_df = pd.DataFrame(con_df['countries'].value_counts())
con_df['country'] = con_df.index
con_df.columns = ['num_movies', 'country']
con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(20)

con_df.loc[con_df.country == 'United States of America', 'num_movies'] = 700
con_df.head(20)
con_df.to_csv('mycsvfile.csv')
data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_movies'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(255, 0,255)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(0,0,0)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Production Countries'),
      ) ]

layout = dict(
    title = 'Production Countries for the Movies (USA is being 700+ to be apple to watch other countries)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
# helper functions to deal with multi-hot features
def group_indices(series,index="id"):
    d={}
    for i in range(series.size):
        l=eval(series.iloc[i])
        for x in l:
            d.setdefault(x[index],[])
            d[x[index]].append(i)
    return d

def get_groups(series,index="name"):
    s=set()
    for i in range(series.size):
        l=eval(series.iloc[i])
        for x in l:s.add(x[index])
    return list(s)

def multi_count(series,index="id"):
    return {k:len(v) for (k,v) in group_indices(series,index).items()}

def expand_multi_feature(df,column,index="id"):
    groups=group_indices(df[column],index=index)
    result=pd.DataFrame()
    for name,indices in groups.items():
        rows=df.iloc[indices].copy()
        rows[column]=name
        result=result.append(rows)
    return result

def multi_groupby(df1,column,index="id"):
    return expand_multi_feature(df,column,index).groupby(column)
# numbers of movies released in each decade
def count_pie(series,filename):
    counts=series.value_counts()
    counts=counts/counts.sum()
    labels=['' if num<0.01 else str(year) for (year,num) in counts.items()]
    f, ax = plt.subplots(figsize=(8, 8))
    explode = [0.02 if counts.iloc[i] < 100 else 0.001 for i in range(counts.size)]
    plt.pie(counts,labels=labels,autopct=lambda x:'{:1.0f}%'.format(x) if x > 1 else '',explode=explode)
    if saving:plt.savefig(filename,dpi=150)
    plt.show()

def count_decade_pie(df,filename):
    count_pie(df2.release_date.dropna().apply(lambda x:str(int(x[:4])//10*10)+'s'),filename)
    
count_decade_pie(df2,filename="pie_decade.png")
# wordcloud of genres and keywords
def multi_wordcloud(series,filename):
    w=wc.WordCloud(background_color="white",margin=20,width=800,height=600,prefer_horizontal=0.7,max_words=50,scale=2)
    count=multi_count(series,"name")
    w.generate_from_frequencies(count)
    if saving:w.to_file(filename)
    f, ax = plt.subplots(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(w)
    plt.show()

multi_wordcloud(df2.genres,filename="wordcloud_genres.png")
multi_wordcloud(df2.keywords,filename="wordcloud_genres2.png")
# distribution of popularity and runtime groupby genres
def plotby_box(df,x,y,filename,yscale="linear"):
    sns.set(style="whitegrid")
    df=df.replace(0,np.nan).copy()
    f,ax=plt.subplots(figsize=(20, 10))
    sns.boxenplot(data=expand_multi_feature(df,x,"name"),x=x,y=y)
    plt.yscale(yscale)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=55,fontsize=20)
    plt.xlabel(x,fontsize=30)
    plt.ylabel(y,fontsize=30)
    if saving:plt.savefig(filename,bbox_inches="tight",dpi=150)
    plt.show()
    
plotby_box(df2,"genres","popularity",yscale="log",filename="genres_popularity.png")
def plotby_bar(df,x,y,filename):
    sns.set(style="whitegrid")
    df=df.replace(0,np.nan).copy()
    f,ax=plt.subplots(figsize=(20, 10))
    sns.barplot(data=expand_multi_feature(df,x,"name"),x=x,y=y)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=55,fontsize=20)
    plt.xlabel(x,fontsize=30)
    plt.ylabel(y,fontsize=30)
    if saving:plt.savefig(filename,bbox_inches="tight",dpi=150)
    plt.show()
    
plotby_bar(df2,"genres","vote_average",filename="genres_vote.png")

# Filter only votes to movies in movies metadata
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
ratings_df = ratings.merge(df2[['id']], left_on=['movieId'], right_on=['id'], how='inner')
# add a new feature, time_dt, to ratings_df by converting timestamp to date
ratings_df['time_dt'] = ratings_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
# split the time_dt to year features
ratings_df['year'] = ratings_df['time_dt'].dt.year
dt = ratings_df.groupby(['year'])['rating'].mean().reset_index()
fig, (ax) = plt.subplots(ncols=1, figsize=(12,5))
plt.plot(dt['year'],dt['rating']);
plt.xlabel('Year');
plt.ylabel('Average ratings');
plt.title('Average ratings per year')
plt.show()
plt.figure(figsize=(10,7))
plt.title('Correlation Matrix')
# mask = np.triu(np.ones_like(md.corr(), dtype=np.bool))
sns.heatmap(df2.corr(),annot=True)
plt.show()
df2.head()
df2.columns
C = df2['vote_average'].mean()
m = df2['vote_count'].quantile(0.9)
C, m
# Filter out movies that don't have 90 % of vote count
q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)

pop = df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6), pop['popularity'].head(6), align='center') 
plt.gca().invert_yaxis()
plt.xlabel('Popularity')
plt.title('Popular Movies')
df2['overview'].head(5)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')

df2['overview'] = df2['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df2['overview'])

tfidf_matrix.shape

cosine_sim = linear_kernel(tfidf_matrix)
# Trying cosine similarity

documents = [
    'alpine snow winter boots.',
    'snow winter jacket.',
    'active swimming briefs',
    'active running shorts',
    'alpine winter gloves'
]

cntvt = CountVectorizer(stop_words='english')

tfidf_matrix = cntvt.fit_transform(documents)
cntvt.get_feature_names()
tfidf_matrix.todense()

cos_sim = cosine_similarity(tfidf_matrix)
cos_sim
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]

idx = indices["The Dark Knight Rises"]
df2['title'].iloc[[i[0] for i in (sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11])]]
get_recommendations('The Dark Knight Rises')
get_recommendations('The Avengers')
# literal_eval is a python function to evaluate correctness of string data. It
# will also create python objects for you 
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']

df2['cast'][0]
df2['crew'][0]
df2['keywords'][0]
df2['genres'][0]

for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
# return top 3
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    
    return []
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
    
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(5)
#data cleaning and prepa

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)
# create "soup" for the vectorization used to compute the cosine similarity matrix

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
get_recommendations('The Dark Knight Rises', cosine_sim2)
get_recommendations('The Godfather', cosine_sim2)
# User-User, Item-Item Collaborative filtering
from surprise import Reader, Dataset, SVD #, cross_validate #evaluate
from surprise.model_selection import cross_validate, KFold
reader=Reader(rating_scale=(1,5))
#read the user rating file (subset file to improve processing time)
ratings=pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# data.split(n_folds=5)

svd= SVD()
# evaluate(svd, data, measures=['RMSE','MAE'])
# cross_validate(NormalPredictor(), data, cv=5)

# Run 5-fold cross-validation and print results
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#create a training set for svd
trainset = data.build_full_trainset()
svd.fit(trainset)
#getting all userId =1 with the rattings
ratings[ratings['userId'] == 1]
str(svd.predict(1, 302).est)
try:  # SciPy >= 0.19
    from scipy.special import comb, logsumexp
except ImportError:
    from scipy.misc import comb, logsumexp  # noqa 
!pip install auto-sklearn
!apt-get remove swig 
!apt-get install swig3.0 build-essential -y
!ln -s /usr/bin/swig3.0 /usr/bin/swig
!apt-get install build-essential
!pip install --upgrade setuptools
# !pip install sklearn

import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import os  
import autosklearn.regression
from sklearn.model_selection import train_test_split
# from scipy.special import comb
# import sklearn
# import sklearn.model_selection
movies = df2.dropna(subset=['vote_average', 'budget', 'revenue'], how='all')
X = movies[['budget', 'revenue']]
y = movies['vote_average']

 

X = X.iloc[:, :].values
y = y.iloc[:].values
y = y.astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_cv_example_tto2',
    output_folder='/tmp/autosklearn_cv_example_oto22',
    delete_tmp_folder_after_terminate=False,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},
)

 

# fit() changes the data in place, but refit needs the original data. We
# therefore copy the data. In practice, one should reload the data
automl.fit(X_train.copy(), y_train.copy(), dataset_name='movie_recommendation')
# During fit(), models are fit on individual cross-validation folds. To use
# all available data, we call refit() which trains all models in the
# final ensemble on the whole dataset.
automl.refit(X_train.copy(), y_train.copy())

 

print(automl.show_models())
predictions = automl.predict(X_test)
print("Accuracy as per AutoML: ", sklearn.metrics.accuracy_score(y_test, predictions))