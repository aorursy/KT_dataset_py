import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import warnings
import collections
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

dataset = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
dataset.head()
dataset.shape
dataset.columns
# generating report
pandas_profiling.ProfileReport(dataset)
# to seperate genre column in dataset
seperated_genres = dataset['Genres'].str.get_dummies(',')

# to concatenate two dataframes
dataset = pd.concat([dataset, seperated_genres], axis = 1, sort = False)
# seperating movies viewing platforms

netflix_movies = dataset.loc[dataset['Netflix'] == 1]
hulu_movies = dataset.loc[dataset['Hulu'] == 1]
prime_video_movies = dataset.loc[dataset['Prime Video'] == 1]
disney_movies = dataset.loc[dataset['Disney+'] == 1]
# dropping columns of other movies watching platforms and unnecessary columns

netflix_movies = netflix_movies.drop(['Hulu', 'Prime Video', 'Disney+', 'Type', 'Unnamed: 0','Genres'], axis = 1)
hulu_movies = hulu_movies.drop(['Netflix', 'Prime Video', 'Disney+', 'Type', 'Unnamed: 0','Genres'], axis = 1)
prime_video_movies = prime_video_movies.drop(['Hulu', 'Netflix', 'Disney+', 'Type', 'Unnamed: 0','Genres'], axis = 1)
disney_movies = disney_movies.drop(['Hulu', 'Prime Video', 'Netflix', 'Type', 'Unnamed: 0','Genres'], axis = 1)
disney_movies.head()
dataset.info()
index_netflix = netflix_movies.index
total_netflix_movies = len(index_netflix)

index_hulu = hulu_movies.index
total_hulu_movies = len(index_hulu)

index_prime = prime_video_movies.index
total_prime_movies = len(index_prime)

index_disney = disney_movies.index
total_disney_movies = len(index_disney)
# Pie chart showing platforms with most number of the movies 

labels = 'Netflix' , 'Hulu', 'Prime Video', 'Disney+'
sizes = [total_netflix_movies,total_hulu_movies,total_prime_movies,total_disney_movies]
explode = (0.1, 0.1, 0.1, 0.1 )

fig1 , ax1 = plt.subplots()

ax1.pie(sizes,
        explode = explode,
        labels = labels,
        autopct = '%1.1f%%',
        shadow = True,
        startangle = 100)

ax1.axis ('equal')
plt.show()
# 

netflix_movies['time'] = netflix_movies['Runtime']
netflix_movies['screenplay'] = netflix_movies['time']/60

hulu_movies['time'] = hulu_movies['Runtime']
hulu_movies['screenplay'] = hulu_movies['time'] / 60

prime_video_movies['time'] = prime_video_movies['Runtime']
prime_video_movies['screenplay'] = prime_video_movies['time'] / 60

disney_movies['time'] = disney_movies['Runtime']
disney_movies['screenplay'] = disney_movies['time'] / 60
# practically a movie is not bearable for more than 5 hours ...
# top 30 runtime movies on Netflix 

top_30_screenplay = netflix_movies.sort_values(by = 'screenplay', ascending = False).head(30)

plt.figure(figsize = (15, 10))
sns.barplot(data = top_30_screenplay, y = 'Title', x = 'screenplay', hue = 'Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Hours')
plt.ylabel('Movie')
plt.title('Top 30 movies by Run Time')

plt.show()
# top 30 runtime movies on Hulu
top_30_screenplay = hulu_movies.sort_values(by = 'screenplay', ascending = False).head(30)
plt.figure(figsize = (15, 10))
sns.barplot(data = top_30_screenplay, y = 'Title', x = 'screenplay', hue = 'Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Hours')
plt.ylabel('Movie')
plt.title('Top 30 movies by Run Time')
plt.show()
# top 30 runtime movies on Amazon Prime Video

top_30_screenplay = prime_video_movies.sort_values(by = 'screenplay', ascending = False).head(30)

plt.figure(figsize = (15, 10))
sns.barplot(data = top_30_screenplay, y = 'Title', x = 'screenplay', hue = 'Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Hours')
plt.ylabel('Movie')
plt.title('Top 30 movies by Run Time')
plt.show()
# top 30 runtime movies on Disney+

top_30_screenplay = disney_movies.sort_values(by = 'screenplay', ascending = False).head(30)

plt.figure(figsize = (25, 15))
sns.barplot(data = top_30_screenplay, y = 'Title', x = 'screenplay', hue = 'Country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Hours')
plt.ylabel('Movie')
plt.title('Top 30 movies by Run Time')
plt.show()
# streaming platform with most movies above 8+ rating (IMDb)
rate_mov_net = netflix_movies['IMDb'] > 8
print("Total Movies on Netflix with more than 8+ rating(IMDb) :",rate_mov_net.sum())
rate_mov_dis = disney_movies['IMDb'] > 8
print("Total Movies on Disney+ with more than 8+ rating(IMDb) :",rate_mov_dis.sum())
rate_mov_pvm = prime_video_movies['IMDb'] > 8
print("Total Movies on amazon prime video with more than 8+ rating(IMDb) :",rate_mov_pvm.sum())
rate_mov_hulu = hulu_movies['IMDb'] > 8
print("Total Movies on Hulu with more than 8+ rating(IMDb) :",rate_mov_hulu.sum())
top_rated = [rate_mov_net.sum(),rate_mov_dis.sum(),rate_mov_pvm.sum(),rate_mov_hulu.sum()]
top_plat = ['Netflix', 'Disney', 'Prime Video', 'Hulu']
top_rated_data = pd.DataFrame({
    'platforms' : ['Netflix', 
                   'Disney', 
                   'Prime Video', 
                   'Hulu'],
    'total_mov' : [rate_mov_net.sum(),
                   rate_mov_dis.sum(),
                   rate_mov_pvm.sum(),
                   rate_mov_hulu.sum()]
})
plt.figure(figsize = (10, 10))
sns.barplot(data = top_rated_data,
           x = top_rated_data['platforms'],
           y = top_rated_data['total_mov']
)
plt.ylabel('Platform')
plt.xlabel('Total number of 8+ rated movies')
plt.title('Platform with most movies rated above 8+ (IMDB)')
plt.show()
list_genre = dataset['Genres'].str.split(',',expand = True)
list_genre
# converting all elements of dataframe into Strings
list_genre = list_genre.applymap(str)
# extracting Genres from the df
genres = []
for i in range(0,9):
    list_genre[i]
    for j in range(0, 16744) :
        if (list_genre[i][j] not in genres) and list_genre[i][j] != 'None' and list_genre[i][j] != 'nan':
            genres.append(list_genre[i][j])
        else:
            pass  
genres
# the following function is used to display top rated movies based on your favourite Genres and respective platforms
def top_rated(genre, platform, n_top):
    genre = platform.loc[platform[genre] == 1]

    top_50 = genre.sort_values(by = 'IMDb', ascending = False).head(n_top)

    plt.figure(figsize = (15, 10))
    sns.barplot(data = top_50, y = 'Title', x = 'IMDb', dodge = False)
    plt.legend(loc = 'lower right')
    plt.xlabel('Ratings', FontSize = 25)
    plt.ylabel('Movies', FontSize = 25)
    plt.title('Top 50 movies by your fav genres',)
    plt.show()
    
top_rated ('Horror',netflix_movies, 50)
top_rated ('Action',hulu_movies, 50)
top_rated ('Sci-Fi',prime_video_movies, 30)
top_rated ('Action',disney_movies, 50)
# Creating Reccomendation system (only for movies and tv_shows available on netflix.)
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df
data = df[['title','director','cast','listed_in','description']]
data.head(3)
# importing required libraries
!pip install rake_nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
rake = Rake()
data['director'] = data['director'].fillna(' ')
data['director'] = data['director'].astype(str)
data['cast'] = data['cast'].fillna(' ')
data['cast'] = data['cast'].astype(str)
data['key_notes'] = ''
for index,row in data.iterrows():
    plot = row['description']
    
    rake.extract_keywords_from_text(plot)
    keyword_score = rake.get_word_degrees()
    
    genre = ''.join(row['listed_in'].split(',')).lower()
    director = ''.join(row['director'].replace(' ','').split(',')).lower()
    cast = ' '.join(row['cast'].replace(' ','').split(',')).lower()
    keyword_score = ' '.join(list(keyword_score.keys()))
    
    row['key_notes'] = genre + ' ' + ' ' + director + ' ' + cast + ' ' + keyword_score

recommend = data[['title','key_notes']]
recommend.head()
cv = CountVectorizer()
count_mat = cv.fit_transform(recommend['key_notes'])
cosine_sim = cosine_similarity(count_mat,count_mat)
print(cosine_sim)
indices = pd.Series(recommend['title'])
def recommend_movie(name):
    movie=[]
    idx = indices[indices == name].index[0]
    sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    recommendation_5= sort_index.iloc[1:5]
    for i in recommendation_5.index:
        movie.append(indices[i])
    return movie
def rec():
    try:
        i = 1
        while(i > 0):
            name = input("Enter The Name of a Movie or Tv Show: ")
            if name.lower() == 'quit':
                break
            else:
                print(recommend_movie(name))

    except KeyboardInterrupt:
        print("The movie or Tv Show does not exist\n")
        rec()

    except IndexError:
        print("The movie or Tv Show does not exist\n")
        rec()
        

print("To exit Enter \"quit\" \n")
rec()
'''Thank you'''