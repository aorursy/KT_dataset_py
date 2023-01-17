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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from plotly.offline import iplot
import cufflinks as cf
cf.go_offline()
import plotly.graph_objects as go
fig = go.Figure()

from wordcloud import WordCloud

import plotly.express as px 
%matplotlib inline
path = '/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv'
data = pd.read_csv(path)
data.head()
cols = data.columns.tolist()
cols
data.drop(['Unnamed: 0','ID',], axis=1, inplace = True)
cols = data.columns.tolist()
cols
data.info()
print("Percentage Missing Data")
(data.isnull().sum()/data.shape[0])*100
data.Age.value_counts()
age_map = {'18+' : 18, '7+' : 7, '13+': 13, 'all' : 0, '16+' : 16}
data['intAge'] = data['Age'].map(age_map)
data.head()
data['Rotten Tomatoes'].value_counts().sort_values(ascending=False)
data['New_Rotten_Tomatoes'] = data['Rotten Tomatoes'].str.replace("%","")
for i in data['New_Rotten_Tomatoes']:
    if i is str:
        i = i.astype(int)
    
data.info()
data['New_Rotten_Tomatoes']
data['Age'].value_counts().iplot('bar', xTitle='Age Group', 
                                        yTitle='Count of Movies', 
                                        title="Number of Movies in specific age group in All services")
data_netflix = data.copy()
data_netflix = data_netflix[data['Netflix']==1]
data_netflix['Age'].value_counts().iplot('bar', colors='Blue', xTitle='Age Group', 
                                        yTitle='Count of Movies', 
                                        title="Number of Movies in specific age group in NetFlix")
data_hulu = data.copy()
data_hulu = data_hulu[data['Hulu']==1]
data_hulu['Age'].value_counts().iplot('bar', colors='Red', xTitle='Age Group', 
                                        yTitle='Count of Movies', 
                                        title="Number of Movies in specific age group in Hulu")
data_prime = data.copy()
data_prime = data_prime[data['Prime Video']==1]
data_prime['Age'].value_counts().iplot('bar', colors='Black', xTitle='Age Group', 
                                        yTitle='Count of Movies', 
                                        title="Number of Movies in specific age group in Prime Video")
data_disney = data.copy()
data_disney = data_disney[data['Disney+']==1]
data_disney['Age'].value_counts().iplot('bar', colors='Purple', 
                                        xTitle='Age Group', 
                                        yTitle='Count of Movies', 
                                        title="Number of Movies in specific age group in Disney+")
data['Rotten Tomatoes'].value_counts().iplot(kind = 'bar', colors = 'Cyan', xTitle = "Ratings", yTitle="Number of Movies", title="Overall Rotten Tomato Ratings")
rotten_tomato_scores = pd.DataFrame({'Streaming Service': ["Prime Video", "Hulu","Disney+","NetFlix"],
                                    'Rotten Tomato Score' : [data_prime['Rotten Tomatoes'].value_counts()[0], 
                                                             data_hulu['Rotten Tomatoes'].value_counts()[0],
                                                             data_disney['Rotten Tomatoes'].value_counts()[0],
                                                             data_netflix['Rotten Tomatoes'].value_counts()[0]]})
rotten_tomato_scores.head()
rotten_tomato_scores.sort_values(ascending=False, by="Rotten Tomato Score").iplot(kind='bar', x='Streaming Service', y='Rotten Tomato Score', 
                           color='Violet', xTitle="Streaming Service", 
                           yTitle="Count of Movies with Score of 100%", 
                           title="Streaming Service with 100% Rotten Tomato Score")
data['IMDb'].value_counts().iplot(kind="bar", color="Red", xTitle='IMDb Ratings', yTitle="Count of Movies", title = "Count of Movies vs IMDb Ratings")
def get_imdb_count_per_service(d):
    # This function returns the number of movies in a service having IMDb score greater than 7.5
    num_of_movies = 0
    for key,value in d.items():
        if key>=7.5:
            num_of_movies+=value
    return num_of_movies
imdb_prime_count = get_imdb_count_per_service(dict(data_prime['IMDb'].value_counts().sort_values(ascending=False)))
imdb_hulu_count = get_imdb_count_per_service(dict(data_hulu['IMDb'].value_counts().sort_values(ascending=False)))
imdb_disey_count = get_imdb_count_per_service(dict(data_disney['IMDb'].value_counts().sort_values(ascending=False)))
imdb_netflix_count = get_imdb_count_per_service(dict(data_netflix['IMDb'].value_counts().sort_values(ascending=False)))

imdb_scores = pd.DataFrame({'Streaming Service': ["Prime Video", "Hulu","Disney+","NetFlix"],
                                    'IMDb Score' :[imdb_prime_count, imdb_hulu_count, imdb_disey_count, imdb_netflix_count] })
imdb_scores.head()
imdb_scores.sort_values(ascending=False, by='IMDb Score').iplot(kind="bar", color="Cyan",x='Streaming Service', 
                  xTitle='Streaming Service', 
                  yTitle="Count of Movies with IMDB Score >=7.5", 
                  title = "Streaming Services with Movies having IMDB >= 7.5")
data['Language'].value_counts()
%%time
languages_dict = dict(data['Language'].value_counts())
languages = set()
for lang,count in languages_dict.items():
    curr_lang = lang
    curr_langs = curr_lang.split(",")
    for i in curr_langs:
        if i in languages:
            continue
        else:
            languages.add(i.lower())
languages = list(languages)
print("Total number of languages are : ", len(languages))
%%time
languages_count = dict()
for lang,count in languages_dict.items():
    curr_lang = lang.split(",")
    for i in curr_lang:
        if i in languages_count.keys():
            languages_count[i] = languages_count.get(i) + 1
        else:
            languages_count[i] = 1
lang_count_df = pd.DataFrame(languages_count.items(), columns=['Language', 'Count'])
lang_count_df.head()
lang_count_df.sort_values(ascending=False, by='Count')[:20].iplot(kind='bar', 
                                                                  x='Language', 
                                                                  xTitle='Language', 
                                                                  yTitle='Count', colors='Green', 
                                                                  title='Language vs Count')
lang_count_df_copy = lang_count_df[:20]
fig = px.pie(values=lang_count_df_copy['Count'], names=lang_count_df_copy['Language']) 
fig.show()
pd.DataFrame(dict(data['Runtime'].value_counts().sort_values(ascending=False)[:20]).items(), columns=['Runtime', 'Count']).iplot(kind='bar' ,
                                                                                                                                 x='Runtime', 
                                                                                                                                 xTitle='Runtime', 
                                                                                                                                 yTitle='Count', title='Runtime vs Count',
                                                                                                                                colors='Magenta')
len(data['Directors'])
directors = list(set(data['Directors']))
directors.pop(0) #TO REMOVE NAN VALUE
len(directors)
new_director = set()
for d in directors:
    curr_d = d.split(",")
    for direc in curr_d:
        if direc in new_director:
            continue
        else:
            new_director.add(direc)
len(new_director)
%%time
new_data = data[data['Directors'].notna()]
directors_count = dict()
direc_in_data = list(new_data['Directors'])
for xdir in direc_in_data:
    curr_dirs = xdir.split(",")
    for xd in curr_dirs:
        if xd in directors_count.keys():
            directors_count[xd] = directors_count.get(xd) + 1
        else:
            directors_count[xd] = 1
directors_count_df = pd.DataFrame(directors_count.items(), columns=['Director', 'Count'])
directors_count_df.sort_values(ascending=False, by='Count').head()
directors_count_df.sort_values(ascending=False, by='Count')[:20].iplot(kind='bar', 
                                                                  x='Director', 
                                                                  xTitle='Director', 
                                                                  yTitle='Count', colors='Blue', 
                                                                  title='Director vs Count')
data[data['Directors']=='Jay Chapman']
temp_data = data[data['Netflix']==1]
temp_data = temp_data[temp_data['Prime Video']==1]
temp_data
#list(temp_data['Title'])
plt.subplots(figsize = (10,10))

wordcloud = WordCloud (
                    background_color = 'white',
                    width = 720,
                    height = 720
                        ).generate(' '.join(temp_data['Title']))
plt.imshow(wordcloud) # image show
plt.axis('off') # to off the axis of x and y
plt.show()
temp_data_nh = data[data['Netflix']==1]
temp_data_nh = temp_data_nh[temp_data_nh['Hulu']==1]
#list(temp_data_nh['Title'])
plt.subplots(figsize = (10,10))

wordcloud = WordCloud (
                    background_color = 'black',
                    width = 720,
                    height = 720
                        ).generate(' '.join(temp_data_nh['Title']))
plt.imshow(wordcloud) # image show
plt.axis('off') # to off the axis of x and y
plt.show()
%%time
genres_unclean = dict(data['Genres'].value_counts())
genres = set()
for g,count in genres_unclean.items():
    curr_g = g.split(",")
    for xg in curr_g:
        if xg in genres:
            continue
        else:
            genres.add(xg)
%%time
count_genres = dict()
for g,count in genres_unclean.items():
    curr_g = g.split(",")
    for xg in curr_g:
        if xg in count_genres.keys():
            count_genres[xg] = count_genres.get(xg)+1
        else:
            count_genres[xg] = 1
count_genres_df = pd.DataFrame(count_genres.items(), columns=['Genre', 'Count'])
count_genres_df.sort_values(ascending=False, by='Count').iplot(kind="bar", x='Genre', xTitle='Genre', yTitle='Count', title='Count of Genres', color='pink')
plt.subplots(figsize = (10,10))

wordcloud_genre = WordCloud (
                    background_color = 'white',
                    width = 720,
                    height = 720
                        ).generate(' '.join(count_genres_df['Genre']))
plt.imshow(wordcloud_genre) # image show
plt.axis('off') # to off the axis of x and y
plt.show()
data_netflix_top = data_netflix[data_netflix['IMDb']>8.5]
data_netflix_top = data_netflix_top[['Title', 'IMDb']].sort_values(ascending=False, by='IMDb')
plt.figure(figsize=(10,8))
sns.barplot(x='IMDb',y='Title',data=data_netflix_top, palette='deep')
plt.title('Top NetFlix Movies')
plt.show()
data_hulu_top = data_hulu[data_hulu['IMDb']>8.5]
data_hulu_top = data_hulu_top[['Title', 'IMDb']].sort_values(ascending=False, by='IMDb')
plt.figure(figsize=(10,8))
sns.barplot(x='IMDb',y='Title',data=data_hulu_top, palette='husl')
plt.title('Top Hulu Movies')
plt.show()
data_disney_top = data_disney[data_disney['IMDb']>8.5]
data_disney_top = data_disney_top[['Title', 'IMDb']].sort_values(ascending=False, by='IMDb')
plt.figure(figsize=(10,8))
sns.barplot(x='IMDb',y='Title',data=data_disney_top, palette='husl')
plt.title('Top Disney+ Movies')
plt.show()
data_prime_top = data_prime[data_prime['IMDb']>8.5]
data_prime_top = data_prime_top[['Title', 'IMDb']].sort_values(ascending=False, by='IMDb')[:10]
plt.figure(figsize=(10,8))
sns.barplot(x='IMDb',y='Title',data=data_prime_top, palette='husl')
plt.title('Top Prime Videos Movies')
plt.show()
dur_n = round(data_netflix['Runtime'].sum()/data_netflix.shape[0],2)
dur_h = round(data_hulu['Runtime'].sum()/data_hulu.shape[0],2)
dur_p = round(data_prime['Runtime'].sum()/data_prime.shape[0],2)
dur_d = round(data_disney['Runtime'].sum()/data_disney.shape[0],2)
print(dur_n, dur_h, dur_p, dur_d)
duration_df = pd.DataFrame({
    'Streaming Platform' : ['NetFlix','Hulu','Prime Video','Disney+'],
    'Duration' : [dur_n, dur_h, dur_p, dur_d]
})
duration_df.head()
duration_df.sort_values(ascending=False, by='Duration').iplot(kind="bar", x='Streaming Platform', 
                                                              xTitle='Streaming Platform', 
                                                              yTitle='Duration', title='Average Duration', color='Red')