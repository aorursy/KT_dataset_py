# Import Libraries

import numpy as np # linear algebra

import pandas as pd # data processing

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import string
#Fetch Data

movies = pd.read_csv('/kaggle/input/imdb-extensive-dataset/IMDb movies.csv')
np.seterr(divide='ignore', invalid='ignore')

movies.drop(83917,inplace=True)
movies.head(2)
movies.columns
plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Movie Ratings Distibution',fontdict=font)

sns.set(style='darkgrid')

ax = movies['avg_vote'].hist(bins=70,color='orange',edgecolor='black')

ax.tick_params(axis='both', colors='darkcyan', labelsize=20)

plt.show()
movie_count = pd.DataFrame()

movie_count['Num Of Movies'] = movies.groupby('country').count()['imdb_title_id'].sort_values(ascending=False).head(10)



movie_count.index.names = ['Country']

movie_count.reset_index(inplace=True)



plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Top Most Movie Making Countries',fontdict=font)

ax = sns.barplot(x='Country',y='Num Of Movies',data = movie_count,palette='gist_rainbow')

ax.xaxis.label.set_color('darkcyan')

ax.yaxis.label.set_color('darkcyan')

ax.xaxis.label.set_size(20)

ax.yaxis.label.set_size(20)

ax.tick_params(axis='both', colors='darkcyan', labelsize=20)

plt.xticks(rotation=45)



#Display the actual values on the bars

for p in ax.patches:

    ax.annotate(format(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()-400), ha = 'center',

                va = 'center', xytext = (0, 10), textcoords = 'offset points',fontweight = 'bold',fontsize=15)
top_prod_cmpny = pd.DataFrame()



top_prod_cmpny['Count'] = movies.groupby('production_company').count()['imdb_title_id'].sort_values(ascending=False).head(10)

top_prod_cmpny.index.names = ['Production Company']

top_prod_cmpny.reset_index(inplace=True)



plt.figure(figsize=(16,10))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

cmap = plt.get_cmap("rainbow")

colors = cmap(np.array([25,50,75,100, 125,150, 175, 200,225, 250]))

textprops = {"fontsize":20,"color":"black"}

plt.title('Movie Distribution among Top 10 Production Companies',fontdict=font,pad=50)

plt.pie(top_prod_cmpny['Count'], labels=top_prod_cmpny['Production Company'], autopct='%1.1f%%', startangle=140,textprops=textprops, colors=colors)

plt.axis('equal')

plt.show()
#Combining all the required info together

df = pd.DataFrame()

movies.fillna(value='',inplace=True)

df['title'] = movies[movies['country']=='USA']['original_title']

df['movie_text_data'] = movies[movies['country']=='USA'].apply(lambda x:x[1]+' '+x[2]+' '+x[5]+' '+x[9]+' '+x[10]+' '+x[11]+' '+x[12]+' '+x[13],axis=1)

df.head()
df.shape
#Method to remove punctuations, stopwords and lower the case

def text_process(movie_text_data):

    #remove punctuations

    nopunc = [char for char in movie_text_data if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word.lower() for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#CountVectorizer will convert a collection of text documents to a matrix of token counts

#And this matrix has lots of zeroes hence called as sparse matrix

count = CountVectorizer(analyzer=text_process).fit(df['movie_text_data'])

sparse_matrix = count.transform(df['movie_text_data'])

sparse_matrix.shape
#Gives the similarity between two non zero vectors of the inner product space

similarity = cosine_similarity(sparse_matrix,sparse_matrix)
df.reset_index(inplace=True)

indices = pd.Series(df.index,index=df['title'])
def get_content_based_recommendation(title,similarity=similarity):

    idx = indices[title]

    

    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(similarity[idx]))    

    

    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)    

    

    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:6]    

    

    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]    

    

    # Return the top 10 most similar movies

    return df['title'].iloc[movie_indices]    
#Fetch Data

movie_lens_movie_info = pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.item",sep="\|",header= None)

movie_lens_rating = pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.data',sep='\t',names=['user_id','item_id','rating','timestamp'])
movie_lens_rating.head()
#Collect the movie titles only, by splitting the year of release info 

movie_lens_movie = pd.DataFrame()

movie_lens_movie[['item_id','title']] = movie_lens_movie_info[[0,1]].copy()

movie_name = movie_lens_movie['title'].str.rsplit(" (",1)

movie_lens_movie['title'] = [movie_name[i][0] for i in range(0,len(movie_name))]

movie_lens_movie.head()
#Collect user ID, movie title and respective rating in a dataframe

user_rating = pd.merge(movie_lens_rating,movie_lens_movie,on='item_id',how='left')

user_rating.drop(['item_id','timestamp'],axis=1,inplace=True)

user_rating.head()
#Number of ratings for each movie and its mean rating

rating = pd.DataFrame(user_rating.groupby('title')['rating'].mean())

rating['num of rating'] = pd.DataFrame(user_rating.groupby('title')['rating'].count())

rating.head()
plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Movie Ratings Distibution',fontdict=font)

sns.set(style='darkgrid')

ax = rating['rating'].hist(bins=70,color='orange',edgecolor='black')

ax.tick_params(axis='both', colors='darkcyan', labelsize=20)
plt.figure(figsize=(16,6))

font = {'color':  'darkcyan',

        'weight': 'bold',

        'size': 30,

        }

plt.title('Number Of Movie Ratings Distibution',fontdict=font)

sns.set(style='darkgrid')

ax = rating['num of rating'].hist(bins=40,color='orange',edgecolor='black')

ax.tick_params(axis='both', colors='darkcyan', labelsize=20)
sns.jointplot(x='rating',y='num of rating',data=rating,alpha=0.5)

plt.show()
#Create a matrix with User ID on one axis and movie title on the other

movie_mat = user_rating.pivot_table(index='user_id',columns='title',values='rating')

movie_mat.head()
def get_collaborative_filtering_recommendation(movie):

    

    #get the rating by each user for the movie input

    user_rating = movie_mat[movie]

    

    #get the correlation with respect to other movies and drop NaN values

    similar_to_movie = movie_mat.corrwith(user_rating)

    corr_with_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])

    corr_with_movie.dropna(inplace=True)

    

    #consider only the movies with 100+ ratings

    corr_with_movie = corr_with_movie.join(rating['num of rating'])

    corr_with_movie = corr_with_movie[corr_with_movie['num of rating']>100].sort_values('Correlation',ascending=False)

    corr_with_movie.reset_index(inplace=True)

    

    #return the top 5 correlated movies

    return corr_with_movie['title'][1:6]
get_content_based_recommendation('Toy Story')
get_content_based_recommendation('Avengers: Infinity War')
get_content_based_recommendation('Kung Fu Panda')
get_collaborative_filtering_recommendation('Star Wars')
get_collaborative_filtering_recommendation('Liar Liar')