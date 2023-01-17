# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
movies = pd.read_csv('/kaggle/input/movies_IMDB.csv')
movies.head()
movies.drop(columns=['Unnamed: 0'] , inplace=True)
movies.head()
movies.describe().T
movies['genre'].value_counts().nlargest(10).plot(kind='pie' ,title= 'Movies per Genre in %', figsize=(10,10), autopct='%1.1f%%',fontsize=15)

movies['year'].value_counts().nlargest(40).plot(kind='bar', figsize=(8
,8))

top_movies = movies.sort_values('imdb',ascending=False).head(50)
fig,ax = plt.subplots(figsize=(30, 7))
# Draw a bar graph
ax = sns.barplot(x='movie_name', y='imdb', data=top_movies,ci=None)
# Rotate the directors' name 45 degrees 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Title the graph
fig.suptitle('Top movies', fontsize=12)
# Set font size of axis label
ax.set_xlabel('movie name',fontsize=20)
ax.set_ylabel('imdb rate',fontsize=20)
# Set tick size of axis 
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

# Show the graph
plt.show();
all_top_movies= movies.sort_values('imdb',ascending=False)
top_2019 = all_top_movies[all_top_movies['year'] == 2019]
fig,ax = plt.subplots(figsize=(30, 7))
# Draw a bar graph
ax = sns.barplot(x='movie_name', y='imdb', data=top_2019,ci=None)
# Rotate the directors' name 45 degrees 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Title the graph
fig.suptitle('Top movies', fontsize=18)
# Set font size of axis label
ax.set_xlabel('movie name',fontsize=20)
ax.set_ylabel('imdb rate',fontsize=20)
# Set tick size of axis 
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
# Show the graph
plt.show();
all_top_metascore_25 = movies.sort_values('metascore',ascending=False).head(25)
fig,ax = plt.subplots(figsize=(30, 7))
# Draw a bar graph
ax = sns.barplot(x='movie_name', y='metascore', data=all_top_metascore_25,ci=None)
# Rotate the directors' name 45 degrees 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Title the graph
fig.suptitle('Top movies bassed on metascore', fontsize=18)
# Set font size of axis label
ax.set_xlabel('movie name',fontsize=20)
ax.set_ylabel('metascore',fontsize=20)
# Set tick size of axis 
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
# Show the graph
plt.show();
all_top_metascore = movies.sort_values('metascore',ascending=False)
top_25metascore_2019 = all_top_metascore[all_top_metascore['year'] == 2019]
fig,ax = plt.subplots(figsize=(30, 7))
# Draw a bar graph
ax = sns.barplot(x='movie_name', y='metascore', data=top_25metascore_2019,ci=None)
# Rotate the directors' name 45 degrees 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Title the graph
fig.suptitle('Top movies bassed on metascore', fontsize=18)
# Set font size of axis label
ax.set_xlabel('movie name',fontsize=20)
ax.set_ylabel('metascore',fontsize=20)
# Set tick size of axis 
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
# Show the graph
plt.show();
all_top_usgross_25 = movies.sort_values('us_grossMillions',ascending=False).head(25)
fig,ax = plt.subplots(figsize=(30, 7))
# Draw a bar graph
ax = sns.barplot(x='movie_name', y='us_grossMillions', data=all_top_usgross_25,ci=None)
# Rotate the directors' name 45 degrees 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Title the graph
fig.suptitle('Top grossMillions movies', fontsize=18)
# Set font size of axis label
ax.set_xlabel('movie name',fontsize=20)
ax.set_ylabel('grossMillions',fontsize=20)
# Set tick size of axis 
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
# Show the graph
plt.show();
all_top_usgross = movies.sort_values('us_grossMillions',ascending=False)
top_2019_gross = all_top_usgross[all_top_usgross['year'] == 2019]
fig,ax = plt.subplots(figsize=(30, 7))
# Draw a bar graph
ax = sns.barplot(x='movie_name', y='us_grossMillions', data=top_2019_gross,ci=None)
# Rotate the directors' name 45 degrees 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Title the graph
fig.suptitle('Top grossMillions movies', fontsize=18)
# Set font size of axis label
ax.set_xlabel('movie name',fontsize=20)
ax.set_ylabel('grossMillions',fontsize=20)
# Set tick size of axis 
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
# Show the graph
plt.show();