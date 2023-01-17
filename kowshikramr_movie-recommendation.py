# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet

from scipy import stats

from ast import literal_eval



from surprise import Reader, Dataset, SVD

from surprise.model_selection import cross_validate

import warnings; warnings.simplefilter('ignore')
sr=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')

sr.head()
sr.shape
np.sum(sr.isna())
sr['genres'] = sr['genres'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

sr = sr.drop([19730, 29503, 35587])
#Number of votes for the movie

number_votes=sr[sr['vote_count'].notnull()]['vote_count']

number_votes
#Average rating of the movie

vote_average=sr[sr['vote_average'].notnull()]['vote_average']

vote_average
#Mean votes across the whole report

C=np.mean(vote_average)

C
m=np.quantile(number_votes,0.9)

m
Reduced_movies=sr[(sr['vote_count']>m) & (sr['vote_count'].notnull())][['title','popularity','genres','vote_count','vote_average']]

Reduced_movies.head()
np.sum(Reduced_movies.isna())

#There is no null values in the final data set
Reduced_movies['Weighted_rating'] = (Reduced_movies['vote_count']/(Reduced_movies['vote_count']+m))*Reduced_movies['vote_average'] + (m/(Reduced_movies['vote_count']+m))*C

Reduced_movies.head()
#Sort the values in descending order

Reduced_movies=Reduced_movies.sort_values('Weighted_rating',ascending=False)

Reduced_movies.head(20)
def simple_recommendation(data,genre='Nothing',percentile=0.9):

    if genre != 'Nothing':

        data=data[data['genres'].apply(lambda x: True if genre in x else False)]

    votes=data[data['vote_count'].notnull()]['vote_count']

    vote_average=data[data['vote_average'].notnull()]['vote_average']

    C=np.mean(vote_average)

    m=np.quantile(votes,percentile)

    data=data[(data['vote_count']>m) & (data['vote_count'].notnull())]

    data['Weighted_rating'] = (data['vote_count']/(data['vote_count']+m))*data['vote_average'] + (m/(data['vote_count']+m))*C

    return data.sort_values('Weighted_rating',ascending=False)
simple_recommendation(sr,'Nothing')[['title','popularity','genres','vote_count','vote_average']].head(20)
simple_recommendation(sr,'Mystery')[['title','popularity','genres','vote_count','vote_average']].head(20)
sr.columns
sample=simple_recommendation(sr,'Nothing',0.70)

sample['overview']=sample['overview'].fillna('')

sample['tagline']=sample['tagline'].fillna('')

sample['description']=(sample['tagline']+' '+sample['overview']).fillna('')
vector=TfidfVectorizer(stop_words='english',analyzer='word',ngram_range=(1,2))

matrix=vector.fit_transform(sample['description'])

matrix.shape
cosine = linear_kernel(matrix, matrix)
data=sample.reset_index()

titles=data['title']

indices = pd.Series(data.index, index=data['title'])
def content_based_recommendation(title,indices=indices,titles=titles,cosine=cosine):

    idx=indices[title]

    sim_scores = list(enumerate(cosine[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices=[i[0] for i in sim_scores[1:26]]

    return titles.iloc[movie_indices]
content_based_recommendation('Inception')
keyword=pd.read_csv('/kaggle/input/the-movies-dataset/keywords.csv')

credits=pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')

keyword['id'] = keyword['id'].astype('int')

credits['id'] = credits['id'].astype('int')

sr['id'] = sr['id'].astype('int')

sr = sr.merge(credits, on='id')

sr = sr.merge(keyword, on='id')

sample=simple_recommendation(sr,'Nothing',0.70)

sample.columns
sample['cast'] = sample['cast'].apply(literal_eval)

sample['crew'] = sample['crew'].apply(literal_eval)

sample['keywords'] = sample['keywords'].apply(literal_eval)
#one of the important crew members we preferred for watching the film are DIRECTORS
def diro(x):

    for i in x:

        if i['job']=='Director':

            return i['name']

    return ''



def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        if len(names) > 3:

            names = names[:3]

        return names

    return []
sample['director']=sample['crew'].apply(diro)

sample['cast']=sample['cast'].apply(get_list)

sample['keywords']=sample['keywords'].apply(get_list)
data = sample

def clean_data(x):

    if isinstance(x,list):

        return [str.lower(i.replace(" ","")) for i in x]

    else:

        if isinstance(x,str):

            #the director is multiplied with 3 to increase its weight as there are 3 actors.

            return [str.lower(x.replace(" ","")) for i in range(3)] 



features=['director','keywords','cast','genres']

for i in features:

    data[i]=data[i].apply(clean_data)

        
data['All']=(data['director'] + data['keywords'] + data['cast'] + data['genres']).apply(lambda x: ' '.join(x))

data.head()[features+['All']]
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

matrix = count.fit_transform(data['All'])

matrix.shape
cosine = cosine_similarity(matrix, matrix)

data = data.reset_index()

titles=data['title']

indices = pd.Series(data.index, index=data['title'])
content_based_recommendation('The Dark Knight',indices,titles,cosine)
content_based_recommendation('Inception',indices,titles,cosine)