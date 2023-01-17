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
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df.head(5)
df.shape
df.columns
pandas_profiling.ProfileReport(df)
movies = df.loc[df['type'] == 'Movie']
movies
tv_shows = df.loc[df['type'] == 'TV Show']
tv_shows
movies['time'] = movies['duration'].str.split(' ',expand = True)[0]
tv_shows['seasons'] = tv_shows['duration'].str.split(' ',expand = True)[0]

movies['time'] = movies['time'].astype(int)
tv_shows['seasons'] = tv_shows['seasons'].astype(int)

movies['screenplay'] = movies['time']/60
index = tv_shows.index
number_of_rows_tv = len(index)
index = movies.index
number_of_rows_movies = len(index)
labels = 'TV Shows', 'Movies'
sizes = [number_of_rows_tv, number_of_rows_movies]
explode = (0.1, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()
# growth in content over years
# Original Release Year of the movies
# some of the oldest movies
# some  of the oldes tv shows
# content from different countries
# movie duration
top_30_screenplay = movies.sort_values(by = 'screenplay', ascending = False).head(30)

plt.figure(figsize = (12, 10))
sns.barplot(data = top_30_screenplay, y = 'title', x = 'screenplay', hue = 'country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Hours')
plt.ylabel('Movie')
plt.title('Top 50 movies by Run Time')
plt.show()
top_30_screenplay = tv_shows.sort_values(by = 'seasons', ascending = False).head(30)

plt.figure(figsize = (12, 10))
sns.barplot(data = top_30_screenplay, y = 'title', x = 'seasons', hue = 'country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total seasons')
plt.ylabel('TV Shows')
plt.title('Top 50 movies by Run Time')
plt.show()
dates = movies['date_added'].astype(str)
movies_added = []
for i in dates:
    y = i[: : -1]
    y = y[0 : 4]
    y = y[ :: -1]
    movies_added.append(y)
movies_added_in_year = []
for i in movies_added:
    if i not in movies_added_in_year:
        movies_added_in_year.append(i)
    else:
        pass

no_of_movies_added = []
for i in movies_added_in_year:
    count = movies_added.count(i)
    no_of_movies_added.append(count)

mov = dict(zip(movies_added_in_year, no_of_movies_added))

mov = collections.OrderedDict(sorted(mov.items()))
mov.pop('nan')
no_of_movies_added = list()
movies_added_in_year = list()
for i in mov.keys():
    movies_added_in_year.append(i)

for j in mov.values():
    no_of_movies_added.append(j)

x = movies_added_in_year
y = no_of_movies_added
plt.plot(x, y)        
plt.plot(x, y, 'bo')
plt.xlabel('Year')
plt.ylabel('Number of movies added')
plt.plot(y)
plt.plot(y, 'r+') 
dates = tv_shows['date_added'].astype(str)
tv_shows_added = []
for i in dates:
    y = i[: : -1]
    y = y[0 : 4]
    y = y[ :: -1]
    tv_shows_added.append(y)

tv_shows_added_in_year = []
for tv_show in tv_shows_added:
    if tv_show not in tv_shows_added_in_year :
        tv_shows_added_in_year.append(tv_show)
    else:
        pass

no_of_tv_shows_added = []
for i in tv_shows_added_in_year:
    count = tv_shows_added.count(i)
    no_of_tv_shows_added.append(count)

tv = dict(zip(tv_shows_added_in_year, no_of_tv_shows_added))

tv = collections.OrderedDict(sorted(tv.items()))
tv.pop('nan')

mov = collections.OrderedDict(sorted(mov.items()))
# mov.pop('nan')
no_of_tv_shows_added = list()
tv_shows_added_in_year = list()
for i in tv.keys():
    tv_shows_added_in_year.append(i)

for j in tv.values():
    no_of_tv_shows_added.append(j)

x = tv_shows_added_in_year
y = no_of_tv_shows_added
plt.plot(x, y)        
plt.plot(x, y, 'bo')
plt.xlabel('Year')
plt.ylabel('Number of TV Shows added')
plt.plot(y)
plt.plot(y, 'r+') 
released = tv_shows['release_year'].astype(str)
date_released = []
for i in released:
    y = i[: : -1]
    y = y[0 : 4]
    y = y[ :: -1]
    date_released.append(y)

year_shows = []
for i in date_released :
    if i not in year_shows:
        year_shows.append(i)
    else:
        pass
counts = []
for i in year_shows:
    count = date_released.count(i)
    counts.append(count)
tv = dict(zip(year_shows,counts))

tv = collections.OrderedDict(sorted(tv.items()))

year_shows = list()
counts = list()

for i in tv.values():
    counts.append(i)
    
for j in tv.keys():
    year_shows.append(j)

x = year_shows
y = counts
plt.figure(figsize = (20,10))
plt.title('TV Shows Release Date vs TV shows added')
plt.plot(x, y)        
plt.plot(x, y, 'bo')
plt.xlabel('Year')
plt.ylabel('Number of TV Shows added')
plt.plot(y)
plt.plot(y, 'r+') 
'''TV Shows Release Date vs TV shows added'''
released = movies['release_year'].astype(str)
date_released = []
for i in released:
    y = i[: : -1]
    y = y[0 : 4]
    y = y[ :: -1]
    date_released.append(y)

movie_years = []
for i in date_released :
    if i not in movie_years:
        movie_years.append(i)
    else:
        pass
counts = []
for i in movie_years:
    count = date_released.count(i)
    counts.append(count)
mov = dict(zip(movie_years, counts))

mov = collections.OrderedDict(sorted(mov.items()))

movie_years = list()
counts = list()

for i in mov.values():
    counts.append(i)

for j in mov.keys():
    movie_years.append(j)

x = movie_years
y = counts
plt.figure(figsize = (30,10))
plt.title('Movies Release Date vs TV shows added')
plt.plot(x, y)        
plt.plot(x, y, 'bo')
plt.xlabel('Year')
plt.ylabel('Number of TV Shows added')
plt.plot(y)
plt.plot(y, 'r+') 
'''Movies Release Date vs TV shows added'''
# most common director
sns.set(context='notebook', style='darkgrid', palette='deep',
        font='sans-serif', font_scale=1, color_codes=True, rc=None)
plt.figure(figsize = (10, 10))
sns.countplot(y = 'director', data = movies, order = movies['director'].value_counts().head(20).index)
plt.show()
hollywood = movies.query('country == "United States"')

bollywood = movies.query('country == "India"') 
data = df[['title','director','cast','listed_in','description']]
data.head(3)
!pip install rake_nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
data['director'] = data['director'].fillna(' ')
data['director'] = data['director'].astype(str)
data['cast']=data['cast'].fillna(' ')
data['cast']=data['cast'].astype('str')
data['entertainment_keys'] = ''
for index,row in data.iterrows():
    plot = row['description']
    r = Rake()
    r.extract_keywords_from_text(plot)
    keyword_score = r.get_word_degrees()
    g = ''.join(row['listed_in'].split(',')).lower()
    d = ''.join(row['director'].replace(' ','').split(',')).lower()
    a = ' '.join(row['cast'].replace(' ','').split(',')).lower()
    k = ' '.join(list(keyword_score.keys()))
    row['entertainment_keys'] = g + ' ' + ' ' + d + ' ' + a + ' ' + k
mydf = data[['title','entertainment_keys']]
mydf.head()
cv = CountVectorizer()
count_mat = cv.fit_transform(mydf['entertainment_keys'])
cosine_sim = cosine_similarity(count_mat,count_mat)
print(cosine_sim)
indices = pd.Series(mydf['title'])
def recommend_movie(name):
    movie=[]
    idx = indices[indices == name].index[0]
    sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10 = sort_index.iloc[1:11]
    for i in top_10.index:
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