import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
df=pd.read_csv('../input/imdb-extensive-dataset/IMDb movies.csv')
data=pd.DataFrame()
c=df['avg_vote'].mean()

m=df['votes'].quantile(0.60)
data=df[df['votes']>=m]
score=[]
v=data['votes'].values
r=data['avg_vote'].values
def weighted_average(v,r,c,m):
    s=((v*r)/(v+m))+((m*c)/(v+m))
    return s
for i in range(len(v)):
    score.append(weighted_average(v[i],r[i],c,m))
data['weighted_score']=score

data=data.sort_values('weighted_score',ascending=False)
data.head()
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
data['description']=data['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(data['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape
tfidf.get_feature_names()[5000:5010]

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
cosine_sim[1]
indices = pd.Series(data.index, index=data['original_title']).drop_duplicates()
indices[:10]
def recommendation(title,cos=cosine_sim):
    idx=indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data['original_title'].iloc[movie_indices]
movie=input('Please enter the movie name:')
recommendation(movie)
