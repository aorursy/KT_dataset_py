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
df_movies=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')
df_movies.head(2)
data=df_movies
data.status.unique()
data.video.unique()
data.spoken_languages.nunique()
data.shape
missingValues=data.isnull().sum()
print(missingValues[missingValues>0])
data=data.drop(['belongs_to_collection','homepage','title'],axis=1)
data.head(2)
links=pd.read_csv('/kaggle/input/the-movies-dataset/links_small.csv')
links.info()
links.head()
links.isnull().sum()
links=links[links['tmdbId'].notnull()]
links.shape
data.loc[19730]
smd=data[data['id'].isin(links)]
smd.shape
data['id'].head()
links.loc[862]
print(missingValues[missingValues>0])
missingValues=data.isnull().sum()
print(missingValues[missingValues>0])
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
data['tagline']=data['tagline'].fillna('')
data['description']=data['overview']+data['tagline']
data['description']=data['description'].fillna('')
data=data.sample(10000)
data.shape

tf=TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=0,stop_words='english')
tfidf_matrix=tf.fit_transform(data['description'])

cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix)
cosine_sim[0]
data.head()
originalTitles=data['original_title']
data.index
indices=pd.Series(data.index,index=data['original_title'])
def get_recommendations(title):
    idx=indices[title]
    sim_scores=list(enumerate(cosine_sim[idx]))
    print('Firtsore-',sim_scores)
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:31]
    movie_indices=[i[0] for i in sim_scores]
    print('movieindx->',movieindices)
    return originalTitles.iloc[movie_indices]
get_recommendations('Jumanji').head(10)