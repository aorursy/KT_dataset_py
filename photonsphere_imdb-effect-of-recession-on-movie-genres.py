# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

%%capture
data = pd.read_csv('../input/imdb.csv',error_bad_lines=False);

movie_data = data[data['type']=='video.movie']
plot.figure(figsize=(11,8))
plot.scatter(movie_data['year'], movie_data['imdbRating']);
reces_years = [2007, 2008, 2009, 2003, 2002, 1991, 1990, 1981, 1980,
               1975, 1974, 1973, 1970, 1969, 1961, 1960, 1954, 1953, 
               1949, 1938, 1937, 1930, 1929, 1928]
remains_summed = pd.DataFrame()

# Count for movies with rating in the dataset in a given year
for i in range(1888,2018):
    if i in reces_years:
        continue
    try:
        remains_summed.at[i,'num_movies_with_rating'] = \
        round(movie_data[movie_data['year'] == i]['year'].count())
    except ValueError:
        continue
# creating column for average movie rating received for a given year
for i in range(1888,2018):
    if i in reces_years:
        continue
    try:
        remains_summed.at[i,'ave_rating_in_the_year'] = \
        round(movie_data[movie_data['year'] == i]['imdbRating'].mean())
    except ValueError:
        continue
        
g = sns.countplot(x='ave_rating_in_the_year',data=remains_summed)
all_years = pd.DataFrame()

# Count for number of movies in a given year
for i in range(1888,2018):
    try:
        all_years.at[i,'num_movies_with_rating'] = \
        round(movie_data[movie_data['year'] == i]['year'].count())
    except ValueError:
        continue
 
#Average movie ratings for movies in a given year
for i in range(1888,2018):
    try:
        all_years.at[i,'ave_rating_in_the_year'] = \
        round(movie_data[movie_data['year'] == i]['imdbRating'].mean())
    except ValueError:
        continue
sns.countplot(x='ave_rating_in_the_year',data=all_years)
genre_list = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'FilmNoir', 'GameShow', 'History', 'Horror', 'Music', 'Musical',
       'Mystery', 'News', 'RealityTV', 'Romance', 'SciFi', 'Short', 'Sport',
       'TalkShow', 'Thriller', 'War', 'Western']
for k in genre_list:
    for i in range(1888,2018):
        if i in reces_years:
            continue
        try:
            remains_summed.at[i,k] = \
            (movie_data[movie_data['year'] == i][k]==1).sum()
        except ValueError:
            continue
# calculating correlations between genres
correlations = remains_summed[genre_list].corr()

# creating mask to plot only lower triangle of the heatmap
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plotting the heatmap
plot.figure(figsize=(20,12))
g = sns.heatmap(correlations*100, fmt='.0f', mask= mask, 
                cbar=False, cmap='coolwarm',annot=True,
                vmax=80);

# saving the heatmap
#g.figure.savefig('non_recess_year_heatmap.png')
#adult genre correlation matrix
k = 9 #number of variables for heatmap
cols = correlations.nlargest(k, 'Action')['Action'].index
#cm = np.corrcoef(remains_summed[cols].values.T)

sns.set(font_scale=1.25)
sns.set_style('white')

mask = np.zeros_like(remains_summed[cols].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

hm = sns.heatmap(remains_summed[cols].corr()*100, cbar=False, annot=True, square=True,
                fmt='.0f', annot_kws={'size': 15},
                 mask=mask)

plot.figure(figsize=(25,18));
g = sns.clustermap(correlations, cmap='coolwarm');
# creating an empty DataFrame
recession_years = pd.DataFrame()

# creating column with movies with rating for a given year
for i in reces_years:
    try:
        recession_years.at[i,'num_movies_with_rating'] = \
        round(movie_data[movie_data['year'] == i]['year'].count())
    except ValueError:
        continue

# creating column with average rating for the movies in a given year
for i in reces_years:
    try:
        recession_years.at[i,'ave_rating_in_the_year'] = \
        round(movie_data[movie_data['year'] == i]['imdbRating'].mean())
    except ValueError:
        continue
g = sns.countplot(x = 'ave_rating_in_the_year', data= recession_years);
# Calculating the total count for genres in which all the
# movies in the year belongs too

for k in genre_list:
    for i in reces_years:
        try:
            recession_years.at[i,k] = \
            (movie_data[movie_data['year'] == i][k]==1).sum()
        except ValueError:
            continue
# creating list of genres to calculate correlation between them 
# excluding 'game show' (has no data for recession years)
genre_list = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'FilmNoir', 'History', 'Horror', 'Music', 'Musical',
       'Mystery', 'News', 'RealityTV', 'Romance', 'SciFi', 'Short', 'Sport',
       'TalkShow', 'Thriller', 'War', 'Western']

# creating correlations for the genres
correlations = recession_years[genre_list].corr()

# creating mask to display only lower left triangle for the heatmap
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plotting the heatmap
plot.figure(figsize=(20,12))
g = sns.heatmap(correlations*100, fmt='.0f', mask=mask,
                cbar=False, cmap='coolwarm', annot=True)