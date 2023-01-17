import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [18, 8]
from IPython.display import YouTubeVideo

YouTubeVideo('hqFHAnkSP2U', width=800, height=450)
rating_df = pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv', parse_dates=['timestamp'], dtype={'userId': 'uint32', 'movieId': 'uint32', 'rating': 'float32'})
movie_df = pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv', dtype={'movieId': 'uint32'})

rating_df.shape, movie_df.shape
rating_df['gave_rating_year'] = rating_df['timestamp'].dt.year
rating_df['gave_rating_month'] = rating_df['timestamp'].dt.month_name().str[:3]

rating_df.drop('timestamp', axis=1, inplace=True)
rating_df['sentiment_analysis'] = rating_df['rating'].map({
    0.5: 'Negative', 1.0: 'Negative', 1.5: 'Negative', 2.0: 'Negative', 2.5: 'Negative',
    3.0: 'Neutral', 3.5: 'Neutral',
    4.0: 'Positive', 4.5: 'Positive', 5.0: 'Positive'
})
rating_df.head(3)
movie_df.head(3)
final_df = pd.merge(rating_df, movie_df, on='movieId', how='inner')  # by default ‘inner’

final_df.shape
final_df.head()
movie_genres = []

for genre in movie_df['genres']:
    for movie in genre.split('|'):
        movie_genres.append(movie)
genre_counts = pd.Series(movie_genres).value_counts()[:18]
from wordcloud import WordCloud

genres_cloud = WordCloud(width=800, height=400, background_color='white', colormap='magma')
genres_cloud.generate_from_frequencies(genre_counts)

plt.imshow(genres_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
top_7_genres = pd.Series(movie_genres).value_counts()[:7]

sns.barplot(y=top_7_genres.index, x=top_7_genres.values, palette='magma').set_title(
        'Top-7 Movie Genres', fontsize=14, weight='bold')

plt.show()
splot1 = sns.countplot(final_df['sentiment_analysis'])

for p in splot1.patches:
                splot1.annotate(format(p.get_height() / final_df['rating'].shape[0] * 100, '.1f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                               rotation=0, ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')
        
plt.xlabel(None)
plt.title('User Sentiment Analysis (%)', fontsize=20, weight='bold')
plt.show()
def plot_progress_year(feature, title):
    rating_progress = final_df.groupby(feature)['userId'].count()
    
    plt.plot(rating_progress, linestyle='-', marker='o', markersize=10)
    plt.title(title, fontsize=16)
    plt.xticks([1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015], rotation=30)
    plt.ylim([0, 2100000])
    plt.show()

plot_progress_year('gave_rating_year', 'Progress rating count per year')
df_index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_progress = final_df.groupby('gave_rating_month').userId.count().reindex(df_index)

plt.plot(month_progress, linestyle='-', color='red', marker='o', markersize=10)
plt.title('Progress rating count by month')
plt.ylim([1250000, 2100000])
plt.show()
genres_str = movie_df['genres'].str.split('|').astype(str)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
tfidf_matrix = tfidf.fit_transform(genres_str)

tfidf_matrix.shape  # banyak karena n-gram (1,2)
# tfidf.get_feature_names()
# Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]
indices = pd.Series(movie_df.index, index=movie_df['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title, similarity=False):
    
    if similarity == False:
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11] # you can change to 20 movies, even more
    
        movie_indices = [i[0] for i in sim_scores]
    
        return pd.DataFrame({'Movie': movie_df['title'].iloc[movie_indices].values})
    
    
    elif similarity == True:
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        
        movie_indices = [i[0] for i in sim_scores]
        similarity_ = [i[1] for i in sim_scores]
        
        return pd.DataFrame({'Movie': movie_df['title'].iloc[movie_indices].values,
                             'Similarity': similarity_})
genre_recommendations('Kung Fu Panda (2008)', similarity=True)
genre_recommendations("Indiana Jones and the Temple of Doom (1984)", similarity=True)
movie_df['movie_release_year'] = movie_df['title'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)
## option 2

def genre_recommendations_2(title, most_recent=False):
    
    if most_recent == False:
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:10] # you can change to 20 movies, even more
    
        movie_indices = [i[0] for i in sim_scores]
    
        return pd.DataFrame({'Movie': movie_df['title'].iloc[movie_indices].values})
    
    
    elif most_recent == True:
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:14]
        
        movie_indices = [i[0] for i in sim_scores]
        
        most_recent_movie = pd.DataFrame({'Movie': movie_df['title'].iloc[movie_indices].values,
                                          'release_year': movie_df['movie_release_year'].iloc[movie_indices].values})
        
        return most_recent_movie.sort_values('release_year', ascending=False).head(10)
genre_recommendations_2('Green Hornet, The (2011)', most_recent=True)
rating_mean = final_df.groupby('title')['rating'].mean().reset_index()
total_rating = final_df.groupby('title')['rating'].count().reset_index()

total_rating_mean = pd.merge(rating_mean, total_rating, on='title', how='inner')
total_rating_mean.rename(columns={'rating_x': 'rating_mean',
                                  'rating_y': 'total_rating'},
                                  inplace=True)

final_df2 = movie_df.merge(total_rating_mean, on='title', how='left').dropna()
## option 3

def genre_recommendations_3(title, best_rating=False):
    
    if best_rating == False:  # sort by total rating
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:14]
    
        movie_indices = [i[0] for i in sim_scores]
        
        most_rating_movie = pd.DataFrame({'Movie': final_df2['title'].iloc[movie_indices].values,
                                          'total_rating': final_df2['total_rating'].iloc[movie_indices].values})
    
        return most_rating_movie.sort_values('total_rating', ascending=False).head(10)
    
    
    elif best_rating == True:  # sort by best rating
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:14]
        
        movie_indices = [i[0] for i in sim_scores]
        
        most_recent_movie = pd.DataFrame({'Movie': final_df2['title'].iloc[movie_indices].values,
                                          'rating_mean': final_df2['rating_mean'].iloc[movie_indices].values})
        
        return most_recent_movie.sort_values('rating_mean', ascending=False).head(10)
genre_recommendations_3('Taken (2008)', best_rating=True)
genre_recommendations_3('Taken (2008)', best_rating=False)