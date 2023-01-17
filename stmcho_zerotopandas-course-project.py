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
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project="zerotopandas-course-project")
#Setup additional libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#Path of the file to read
netflix_filepath = "/kaggle/input/netflix-shows/netflix_titles.csv"

#Read the file into a dataframe
netflix_raw = pd.read_csv(netflix_filepath, index_col='show_id')
netflix_raw.head()
netflix_raw.info()
netflix_data = netflix_raw.drop(columns=['cast', 'country'])
from datetime import datetime

netflix_data['month_added'] = pd.to_datetime(netflix_data['date_added']).dt.strftime('%B')

netflix_data['year_added'] = netflix_data['date_added'].str[-4:]
netflix_data['year_added'] = pd.to_numeric(netflix_data['year_added'], errors='coerce')
netflix_data['date_added'] = pd.to_datetime(netflix_data['date_added'])
netflix_data['date_added']
movie_data = netflix_data[netflix_data['type']=='Movie']
show_data = netflix_data[netflix_data['type']=='TV Show']
#Change the format of duration in shows
show_data['duration'] = show_data['duration'].str.replace(' Season', '')
show_data['duration'] = show_data['duration'].str.replace('s', '')
show_data['duration'] = show_data['duration'].astype(int)
#Change the format of duration in movies
movie_data['duration'] = movie_data['duration'].str.replace(' min', '')
movie_data['duration'] = movie_data['duration'].astype(int)
#Setup style
sns.set_style('darkgrid')
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['figure.facecolor'] = '#00000000'
ax = sns.countplot(x='type', data=netflix_data);

plt.xlabel('')
plt.ylabel('Count')
plt.title("Amount of Movies vs. TV Shows in Netflix");
movie_amt = movie_data.shape[0]
show_amt = show_data.shape[0]

print("There are {} movies in Netflix, or {:.2%} of the content.".format(movie_amt, movie_amt/(movie_amt+show_amt)))
print("There are {} TV shows in Netflix, or {:.2%} of the content.".format(show_amt, show_amt/(movie_amt+show_amt)))
movie_yearly = movie_data.copy()
show_yearly = show_data.copy()
movie_yearly = movie_yearly['year_added'].value_counts().reset_index()
movie_yearly = movie_yearly.rename(columns = {'year_added': 'count', 'index' : 'year_added'})

show_yearly = show_yearly['year_added'].value_counts().reset_index()
show_yearly = show_yearly.rename(columns = {'year_added': 'count', 'index' : 'year_added'})

new_row = [{'year_added': 2009, 'count': 0},
           {'year_added': 2010, 'count': 0},
           {'year_added': 2011, 'count': 0}
          ]

for row in new_row:
    show_yearly = show_yearly.append(row, ignore_index=True)
#Reformat the years into int
show_yearly['year_added'] = show_yearly['year_added'].astype(int)
movie_yearly['year_added'] = movie_yearly['year_added'].astype(int)
movie_yearly = movie_yearly.where(movie_yearly['year_added']!=2020)
movie_yearly = movie_yearly.dropna()
show_yearly = show_yearly.where(show_yearly['year_added']!=2020)
show_yearly = show_yearly.dropna()
#Sort by years
movie_yearly = movie_yearly.sort_values('year_added')
show_yearly = show_yearly.sort_values('year_added')

#Redo the indices for organization
movie_yearly.reset_index(drop=True, inplace=True)
show_yearly.reset_index(drop=True, inplace=True)
#Percentage of content
movie_yearly['percent'] = movie_yearly['count'].apply(lambda x: 100*x/sum(movie_yearly['count']))
show_yearly['percent'] = show_yearly['count'].apply(lambda x: 100*x/sum(show_yearly['count']))
plt.bar(show_yearly['year_added'], show_yearly['count'], color='r')
plt.bar(movie_yearly['year_added'], movie_yearly['count'], bottom=show_yearly['count']);

plt.title("Netflix Content added over the Years");
show_yearly['growth'] = 0

for year in range(11):
    if(show_yearly.iloc[year, 1] != 0):
        show_yearly.iloc[year+1, 3] = show_yearly.iloc[year+1, 1]/show_yearly.iloc[year, 1]
    
show_yearly.iloc[0, 3] = np.nan
movie_yearly['growth'] = 0

for year in range(11):
    if(movie_yearly.iloc[year, 1] != 0):
        movie_yearly.iloc[year+1, 3] = movie_yearly.iloc[year+1, 1]/movie_yearly.iloc[year, 1]
        
movie_yearly.iloc[0, 3] = np.nan
sns.lineplot(x='year_added', y='growth', data=show_yearly, label='TV Shows')
sns.lineplot(x='year_added', y='growth', data=movie_yearly, label='Movies')

plt.legend()
plt.xlabel("Year")
plt.ylabel("Growth as % of Prior Year")
plt.title("Content Growth in Netflix Annually");
sns.distplot(a=movie_data['duration'])

plt.xlabel("Duration (mins)")
plt.title("Distribution of Movie Duration");
sns.countplot(data=show_data, x='duration')

plt.title('Length of Netflix TV Shows')
plt.xlabel('# of Seasons');
jovian.commit(project="zerotopandas-course-project", privacy='private')
shows_genres = show_data.copy()

#Turn the genres of listed_in into a list to enable one hot encoding
shows_genres['listed_in'] = shows_genres['listed_in'].str.split(', ')
shows_genres['listed_in']
#Iterate through the list of genres and place a 1 in the corresponding column for every row in the dataframe
for index, row in shows_genres.iterrows():
    for genre in row['listed_in']:
        shows_genres.loc[index, genre] = 1

#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
shows_genres = shows_genres.fillna(0)
shows_genres.columns[11:33]
shows_genres[shows_genres['TV Shows']==1]
shows_genres = shows_genres.drop(index=shows_genres[shows_genres['TV Shows']==1].index)
shows_genres = shows_genres.drop(columns='TV Shows')
#Reformat the years into int from float
shows_genres['year_added'] = shows_genres['year_added'].astype(int)
#Sort the top 10 shows with the most seasons
shows_top10 = shows_genres.sort_values('duration', ascending=False).head(10)
shows_top10.plot(kind='bar', x='title', y='duration', legend=False)

plt.xlabel('')
plt.ylabel('# of Seasons')
plt.title("Netflix Shows with Most Seasons");
#Obtain a list of all the genres
genres = []

for col in range(11, 32):
    genres.append(shows_genres.columns[col])
#Get only the genres one-hot encoding for the top 10 shows
top10_genres = shows_top10[genres].reset_index(drop=True)
top10_genres = top10_genres.transpose()
#Get a sum of each genre in the top 10 shows
top10_genres['count'] = top10_genres.sum(axis=1)

#Obtain the genre as a percentage of total genres represented in the top 10 shows
top10_genres['percentage'] = top10_genres['count']/(top10_genres.to_numpy().sum())

#Sort by decreasing genre count
top10_genres = top10_genres.sort_values(['percentage', 'count'], ascending=False)
top10_genres
print("Out of the top 10 shows:\n")

for genre in top10_genres.index:
    print("{} is {:.2%}.".format(genre, top10_genres.loc[genre, 'percentage']))
shows_top10['rating'].value_counts()
shows_genres_yearly = shows_genres.groupby('year_added')[genres].sum()
shows_genres_yearly
shows_genres_yearly.drop(index=[0, 2020], inplace=True)
plt.figure(figsize=(16, 8))

sns.heatmap(shows_genres_yearly, fmt='g', annot=True, cmap='Blues')

plt.ylabel('Year')
plt.title('Heatmap of Netflix TV Show Genres added Annually');
jovian.commit(project="zerotopandas-course-project")
netflix_data[netflix_data['year_added'].notna()==False]
#Drop the rows with NaT in date added
netflix_withdates = netflix_data.copy()
netflix_withdates = netflix_withdates.dropna(subset=['year_added'])

#Reformat the year_added column to int for easier comparison
netflix_withdates['year_added'] = netflix_withdates['year_added'].astype(int)
#Calculate how long from content release did it get added to Netflix
netflix_withdates['release_diff'] = netflix_withdates['year_added'] - netflix_withdates['release_year']
netflix_withdates['release_diff'].unique()
netflix_withdates[netflix_withdates['release_diff']<0]
#Remove the rows with release years that're before theirs year added to Netflix
netflix_withdates = netflix_withdates[netflix_withdates['release_diff']>=0]
temp_df = netflix_withdates[['release_diff', 'type']].value_counts().reset_index()
temp_df = temp_df.rename({0: 'count'}, axis='columns')

#Plot the graph
ax = sns.lineplot(data=temp_df, x='release_diff', y='count', hue='type')
ax.set(xticks=range(0, 100, 10), xlabel='Year(s)', title='Time between Content Release and Netflix Addition')
plt.show()
#Create a list of months in the right order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

#Get the content count by month/year
#Also remove the content from year 2020 since it's incomplete
temp2 = netflix_withdates.groupby('year_added')['month_added'].value_counts().drop(level='year_added', labels=2020).unstack().fillna(0)[month_order]
plt.figure(figsize=(16, 8))

ax = sns.heatmap(temp2, fmt='g', annot=True, cmap='Blues')
ax.set(title='Heatmap of Netflix Content added Monthly', xlabel='', ylabel='')

plt.show()
jovian.commit(project="zerotopandas-course-project")
netflix_directors = netflix_data[netflix_data['director'].notna()]

#Turn the genres of listed_in into a list to enable one hot encoding
netflix_directors['director'] = netflix_directors['director'].str.split(', ')
netflix_directors['director']
#Iterate through the list of genres and place a 1 in the corresponding column for every row in the dataframe
for index, row in netflix_directors.iterrows():
    for director in row['director']:
        netflix_directors.loc[index, director] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
netflix_directors = netflix_directors.fillna(0)
#Checking for any director duplicates
if (netflix_directors.columns.duplicated(keep=False).any() ==True):
    print('there are duplicates')
else: print('nope')
directors_df = netflix_directors.iloc[:,11:].transpose()
directors_df['count'] = directors_df.sum(axis=1)
directors_df = directors_df.sort_values('count', ascending=False)
directors_df.head(10)
print("List of top 15 directors with the most content on Netflix:\n")

for name in range(15):
    print("The #{} director with the most content is {}, with {} works."
         .format(name+1, directors_df.index[name], directors_df.iloc[name, 4265]))
netflix_directors[netflix_directors['Jan Suter']==1]
netflix_directors[netflix_directors['Johnnie To']==1]
jovian.commit(project="zerotopandas-course-project")
#Path of the file to read
rt_filepath = "/kaggle/input/rotten-tomato-movie-reviwe/rotten tomato movie reviwe.csv"

#Read the file into a dataframe
rt_raw = pd.read_csv(rt_filepath)
#Changing the column name to match the movie title column of my first dataset
#Also to make some of them shorter
rt_raw = rt_raw.rename(columns={'Name': 'title', 'TOMATOMETER score': 't_score', 'TOMATOMETER Count': 't_count', 'AUDIENCE score': 'a_score', 'AUDIENCE count':'a_count'})

#Making sure the format of the titles are strings for easy comparisons
rt_raw['title'] = rt_raw['title'].astype(str)

#Reformatting AUDIENCE count as ints for easy comparisons
rt_raw['a_count'] = rt_raw.a_count.astype(str).str.strip().str.replace(',','').astype(int)

#Reformatting Studio names
rt_raw['Studio'] = rt_raw['Studio'].str.strip()
rt_raw.info()
temp_movie = movie_data.set_index(keys='title')
temp_rt = rt_raw.set_index(keys='title')
joint_df = pd.merge(temp_movie, temp_rt, how='inner', on='title')
print("Old dataset volume: {} vs.\nNew dataset volume: {}."
      .format(movie_data.shape, joint_df.shape))
joint_df.head()
#Dropping redundant and irrelevant columns
joint_df = joint_df.drop(columns=['type', 'description', 'Rating', 'Directed By', 'Runtime'])

#Removing the content with less than 100 ratings since those wouldn't be too accurate
joint_df = joint_df[(joint_df.t_count >= 100) | (joint_df.a_count >= 100)]
joint_df.head()
#one hot encoding for studios
studio_df = joint_df[['Studio']]

#Iterate through the studio names to add a 1 to each corresponding studio column
for index, row in studio_df.iterrows():
    studio_df.loc[index, row] = 1
        
#Filling in the NaN values with 0
studio_df = studio_df.fillna(0)

#Drop the Studio column since it's unneeded
studio_df = studio_df.drop(columns='Studio')
studio_df = studio_df.transpose()
studio_df['count'] = studio_df.sum(axis=1)
studio_df = studio_df.sort_values(by='count', ascending=False)
print("The studios with the most content on Netflix: \n")

for rank in range(10):
    print("#{} is {}, with {} works ({:.2%})."
         .format(rank+1, studio_df.index[rank], int(studio_df.iloc[rank, 374]), (studio_df.iloc[rank, 374]/studio_df.shape[1])))
jovian.commit(project="zerotopandas-course-project")
jovian.commit(project="zerotopandas-course-project")