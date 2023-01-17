import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel
movies=pd.read_csv('../input/movies.csv')

ratings=pd.read_csv('../input/ratings.csv')
movies.info()
ratings.info()
movies.shape
ratings.shape
movies.describe()
ratings.describe()
genres=[]

for genre in movies.genres:

    

    x=genre.split('|')

    for i in x:

         if i not in genres:

            genres.append(str(i))

genres=str(genres)    

movie_title=[]

for title in movies.title:

    movie_title.append(title[0:-7])

movie_title=str(movie_title)    
wordcloud_genre=WordCloud(width=1500,height=800,background_color='black',min_font_size=2

                    ,min_word_length=3).generate(genres)

wordcloud_title=WordCloud(width=1500,height=800,background_color='cyan',min_font_size=2

                    ,min_word_length=3).generate(movie_title)
plt.figure(figsize=(30,10))

plt.axis('off')

plt.title('WORDCLOUD for Movies Genre',fontsize=30)

plt.imshow(wordcloud_genre)

plt.figure(figsize=(30,10))

plt.axis('off')

plt.title('WORDCLOUD for Movies title',fontsize=30)

plt.imshow(wordcloud_title)
df=pd.merge(ratings,movies, how='left',on='movieId')

df.head()
df1=df.groupby(['title'])[['rating']].sum()

high_rated=df1.nlargest(20,'rating')

high_rated.head()
plt.figure(figsize=(30,10))

plt.title('Top 20 movies with highest rating',fontsize=40)

colors=['red','yellow','orange','green','magenta','cyan','blue','lightgreen','skyblue','purple']

plt.ylabel('ratings',fontsize=30)

plt.xticks(fontsize=25,rotation=90)

plt.xlabel('movies title',fontsize=30)

plt.yticks(fontsize=25)

plt.bar(high_rated.index,high_rated['rating'],linewidth=3,edgecolor='red',color=colors)

df2=df.groupby('title')[['rating']].count()

rating_count_20=df2.nlargest(20,'rating')

rating_count_20.head()
plt.figure(figsize=(30,10))

plt.title('Top 20 movies with highest number of ratings',fontsize=30)

plt.xticks(fontsize=25,rotation=90)

plt.yticks(fontsize=25)

plt.xlabel('movies title',fontsize=30)

plt.ylabel('ratings',fontsize=30)



plt.bar(rating_count_20.index,rating_count_20.rating,color='red')
cv=TfidfVectorizer()

tfidf_matrix=cv.fit_transform(movies['genres'])
movie_user = df.pivot_table(index='userId',columns='title',values='rating')

movie_user.head()


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices=pd.Series(movies.index,index=movies['title'])

titles=movies['title']

def recommendations(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:21]

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]
recommendations('Toy Story (1995)')