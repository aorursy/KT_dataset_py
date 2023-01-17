# Importing all necessary libraries.
import pandas as pd
import numpy as np
import requests
import time
import json
import unidecode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import warnings

# Supressing future warnings that are irrelevant in our case.
warnings.filterwarnings('ignore')

# Displaying all graphs in the notebook.
%matplotlib inline
# Reading the dataset into a dataframe. We only need some of the columns.
netflix = pd.read_csv('../input/netflix-shows/netflix_titles.csv', usecols=['type', 'title', 'country', 'release_year', 'rating', 'duration'])

# Printing the number of total productions in the dataframe and the proportions of movies and TV shows in %.
production_count = netflix['type'].value_counts(normalize=True)
print('Total productions: {}\n'.format(netflix.shape[0]))
print('Type of production\n{}: {:.2f}%\n{}: {:.2f}%'.format(production_count.index[0], production_count[0] * 100, production_count.index[1], production_count[1] * 100))
# Splitting the countries into a list and selecting only the first country as the primary country of production.
netflix['country'] = netflix['country'].str.split(', ').fillna('no country').apply(lambda x: x[0])
# Doing some additional cleaning and replacing rows with no countriies with missing values.
netflix['country'] = netflix['country'].str.replace(',', '').str.replace('West Germany', 'Germany').replace('n', np.nan)

# Taking a look at the first few rows of the dataset.
netflix.head()
# Grouping the dataset by country and counting the productions for each.
netflix_top_countries = netflix.groupby('country').agg({'title':'count'})
netflix_top_countries.columns = ['count']
netflix_top_countries = netflix_top_countries.sort_values(by='count', ascending=False)
# Displaying the top 10 countries.
print(netflix_top_countries.head(10))
# Writing a function to fetch the IMDb IDs of the Netflix titles.
def get_imdb_ids():
    # Creating an empty list to hold the IDs.
    imdb_ids = []
    # Iterating over every title.
    for i, row in netflix.iterrows():
        title = row['title']

        # Making the API request.
        url = 'https://imdb-internet-movie-database-unofficial.p.rapidapi.com/film/{}'.format(title)
        headers = {
        'x-rapidapi-host': 'insert-host',
        'x-rapidapi-key': 'insert-key'
        }
        response = requests.request("GET", url, headers=headers)
        # Populating the list depending on the response.
        try:
            id_code = response.json()['id']
        except (KeyError, json.JSONDecodeError):
            imdb_ids.append('error')
        else:
            imdb_ids.append(response.json()['id'])

        # A cooldown of 30 seconds every 30 requests, so that we don't overwhelm the server.
        if i in [i for i in range(30, 7000, 30)]:
            time.sleep(10)

    # Saving the list with the IDs as a .txt file.
    with open('imdb_ids.txt', 'w') as f:
        json.dump({'imdb_id': imdb_ids}, f)
        
# get_imdb_ids()

# Reading the IDs into a dataframe.
with open('../input/imdb-data/imdb_ids.txt', 'r') as f:
    imdb_ids = pd.DataFrame(json.load(f))
# Reading the IMDb dataset, we only need the IDs (tconst), the title and the year (which we'll later use just for the movies).
imdb_titles = pd.read_csv('../input/imdb-data/title.basics.tsv', sep='\t', usecols=['tconst', 'primaryTitle', 'startYear'])
imdb_titles.columns = ['imdb_id', 'imdb_title', 'imdb_year']
# Merging the IDs that we retrieved via the API with the IMDB dataset. It's a left merge because we only need to keep the Netflix titles, not the entire IMDB catalogue.
imdb = imdb_ids.merge(imdb_titles, how='left', on='imdb_id')

# Now, let's combine the Netflix and the IMDb datasets together. They're the same length, so this will be easy.
netflix_imdb = pd.concat([netflix, imdb], axis=1)
# Removing empty titles (those for which we couldn't retrieve the ID earlier) and resetting the index.
netflix_imdb = netflix_imdb[netflix_imdb['imdb_title'].notnull()].reset_index(drop=True)

# Displaying the first few rows.
netflix_imdb.head()
# Now, let's combine the Netflix and the IMDb datasets together. They're the same length, so this will be easy.
netflix_imdb = pd.concat([netflix, imdb], axis=1)
# Removing empty titles (those for which we couldn't retrieve the ID earlier) and resetting the index.
netflix_imdb = netflix_imdb[netflix_imdb['imdb_title'].notnull()].reset_index(drop=True)

# Displaying the first few rows.
netflix_imdb.head()
# Removing movies for which the years don't match, and resetting the index.
movie_year_bool = (netflix_imdb['type'] == 'Movie') & (netflix_imdb['release_year'] == netflix_imdb['imdb_year'])
netflix_imdb = netflix_imdb[~movie_year_bool].reset_index(drop=True)

# Decode both the Netflix and IMDb titles, so that some unicode characters (e.g. letters with accents) are concerted to their ASCII alternatives.
netflix_imdb['netflix_title'] = netflix_imdb['title'].apply(lambda x: unidecode.unidecode(x))
netflix_imdb['imdb_title'] = netflix_imdb['imdb_title'].apply(lambda x: unidecode.unidecode(x))
# Removing special characters and punctuation marks from both titles and converting the titles to lowercase.
netflix_imdb['netflix_title'] = netflix_imdb['netflix_title'].str.replace('[?;:!&.\(\),/#$]', '').str.lower()
netflix_imdb['imdb_title'] = netflix_imdb['imdb_title'].str.replace('[?;:!&.\(\),/#$]', '').str.lower()
# Isolating all titles that are not an exact match.
titles_not_matching = netflix_imdb[netflix_imdb['netflix_title'] != netflix_imdb['imdb_title']]

# Splitting the titles into lists of words.
titles_not_matching['netflix_title'] = titles_not_matching['netflix_title'].str.split()
titles_not_matching['imdb_title'] = titles_not_matching['imdb_title'].str.split()
# Defining the function for the title matching.
def percentage_match(df):
    counter = 0
    words_netflix_title = list(set(df['netflix_title']))
    words_imdb_title = list(set(df['imdb_title']))
    for word_netflix in words_netflix_title:
        for word_imdb in words_imdb_title:
             if word_netflix == word_imdb:
                counter += 1
    if len(words_netflix_title) >= len(words_imdb_title):
        pct_match = counter / len(words_imdb_title)
    else:
        pct_match = counter / len(words_netflix_title)
    return pct_match

# Applying the function on the dataset.
titles_not_matching['pct_match'] = titles_not_matching.apply(percentage_match, axis=1)
# Removing those productions for which the match is less than .75.
titles_to_remove = titles_not_matching[titles_not_matching['pct_match'] < .75]
netflix_imdb = netflix_imdb.drop(titles_to_remove.index, axis=0).reset_index(drop=True)

# Finally, let's also remove those productions, fir which the IMDB IDs are still the same. We can't know which one we really need, so we'll simply remove all that duplicate.
netflix_imdb = netflix_imdb.drop_duplicates('imdb_id', keep=False).reset_index(drop=True)
# Reading the IMDB dataset with the scores into a dataframe.
imdb_scores = pd.read_csv('../input/imdb-data/title.ratings.tsv', sep='\t', usecols=['tconst', 'averageRating'])
imdb_scores.columns = ['imdb_id', 'imdb_score']

# Merging the grand dataframe with the scores.
netflix_imdb = netflix_imdb.merge(imdb_scores, how='left', on='imdb_id')
# Removing all titles, for which there are not IMDB scores for some reason (e.g. the title is not yet released).
netflix_imdb = netflix_imdb[netflix_imdb['imdb_score'].notnull()]

# Dropping columns that we don't need anymore.
netflix_imdb.drop(['imdb_id', 'imdb_title', 'netflix_title'], axis=1, inplace=True)

# Displaying the first few rows.
netflix_imdb.head()
# Printing the number of total productions in the dataframe and the proportions of movies and TV shows in %, as well as the % of data lost since the beginning of thia project.
production_count = netflix_imdb['type'].value_counts(normalize=True)
print('Total productions: {}'.format(netflix_imdb.shape[0]))
print('Data lost: {:.2f}%\n'.format((netflix.shape[0] - netflix_imdb.shape[0]) / netflix.shape[0] * 100))
print('Type of production\n{}: {:.2f}%\n{}: {:.2f}%'.format(production_count.index[0], production_count[0]*100, production_count.index[1], production_count[1] * 100))
# Grouping the dataset by country and counting the productions for each.
netflix_reduced_top_countries = netflix_imdb.groupby('country').agg({'title':'count'})
netflix_reduced_top_countries.columns = ['count']
netflix_reduced_top_countries = netflix_reduced_top_countries.sort_values(by='count', ascending=False)
# Displaying the top 10 countries.
print(netflix_reduced_top_countries.head(10))
# Importing latituted and longitude and merging them with the main dataframe.
countries_coordinates = pd.read_csv('../input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv', usecols=['latitude', 'longitude', 'country'])
netflix_top_countries = netflix_top_countries.merge(countries_coordinates, how='left', on='country')

# Adding the mean IMDB scores per country.
netflix_imdb_top_countries = netflix_imdb.groupby('country').agg({'imdb_score':'mean'})
netflix_top_countries = netflix_top_countries.merge(netflix_imdb_top_countries, how='left', on='country')

# We need a dataframe with the TOP 10 countries only.
netflix_top10_countries = netflix_top_countries.head(10)

# Displaying the countries.
netflix_top10_countries
# Generating the map. Specifying the size of the figure.
plt.figure(figsize=(30,20), facecolor='#A9A9A9')

# Setting up parameters of the actual map.
m = Basemap(llcrnrlon=-140, llcrnrlat=-60, urcrnrlon=160, urcrnrlat=90)
m.drawmapboundary(fill_color='None', linewidth=0)
m.fillcontinents(color='black', alpha=1, lake_color='#A9A9A9')
m.drawcountries(linewidth=.5, color='#A9A9A9')

# Iterating over the countries.
for i, row in netflix_top10_countries.iterrows():
    # Adjusting the location of the labels, so that they don't overlap.
    if row['country'] == 'United States':
        x1, y1 = m(row['longitude'] + 8, row['latitude'])
        x2, y2 = m(row['longitude'] - 20, row['latitude'])
    elif row['country'] == 'South Korea':
        x1, y1 = m(row['longitude'] - 8, row['latitude'] - 5)
        x2, y2 = m(row['longitude'] - 15, row['latitude'])
    elif row['country'] == 'Japan':
        x1, y1 = m(row['longitude'] + 5, row['latitude'])
        x2, y2 = m(row['longitude'], row['latitude'] + 6)
    elif row['country'] == 'Turkey':
        x1, y1 = m(row['longitude'] + 5, row['latitude'])
        x2, y2 = m(row['longitude'], row['latitude'] - 6)
    else:
        x1, y1 = m(row['longitude'] + 5, row['latitude'])
        x2, y2 = m(row['longitude'] - 15, row['latitude'])
        
    # Adding the numbers of productions (text1) and IMDb scores (text2).
    text1 = '{}: {:,.0f}'.format(row['country'], row['count'], row['imdb_score'])
    plt.text(x1, y1, s=text1, fontsize=25, c='white', bbox=dict(facecolor='#E50914', alpha=0.5, edgecolor='None'), fontweight='bold')
    text2 = '{:,.2f}'.format(row['imdb_score'])
    plt.text(x2, y2, s=text2, fontsize=25, c='white', bbox=dict(facecolor='purple', alpha=0.5, edgecolor='None'), fontweight='bold')
    
# Adding circles with the size of the number of productions.
m.scatter(netflix_top10_countries['longitude'].to_list(), netflix_top10_countries['latitude'].to_list(), latlon=True, s=netflix_top10_countries['count'].to_list(), alpha=.9, zorder=4, c='#E50914')

# Adding semi-transparent patches behind the labels.
count = mpatches.Patch(color='#E50914', label='Number of productions')
scores = mpatches.Patch(color='purple', label='Average IMDb score')
legend = plt.legend(handles=[count, scores], loc='lower left', fontsize=30, framealpha=0.5)

# Adding a title.
plt.title('Netflix US catalogue in 2019\nTOP 10 countries with the most productions (Movies, TV shows etc.)', fontsize=35, c='white', fontweight='bold')

# Displaying the plot.
plt.show()
# Creating a new column with the age ratings as numbers.
netflix_imdb['rating_number'] = netflix_imdb['rating'].replace(['G', 'TV-Y','PG', 'TV-Y7', 'TV-Y7-FV', 'PG-13', 'TV-G', 'R', 'TV-PG', 'NC-17', 'TV-14', 'TV-MA', 'NR', 'UR'], value=[1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, np.nan, np.nan])
# Updating the type column to indicate whether a movie is a TV movie.
netflix_imdb['type_new'] = np.select([(netflix_imdb['rating'].str.startswith('TV', na=False)) & (netflix_imdb['type'] == 'Movie'), (~netflix_imdb['rating'].str.startswith('TV', na=False)) & (netflix_imdb['type'] == 'Movie')], ['TV movie', 'Cinema movie'], default='TV show')

# Printing the count for each type.
print(netflix_imdb['type_new'].value_counts())

# Formatting the duration column for the movies (in minutes) and converting it to float. We don't need to do that for the TV shows because they have season, and all seasons are not always indicated.
netflix_imdb['duration_min_movie'] = np.select([netflix_imdb['type'] == 'Movie'], [netflix_imdb['duration']], default=np.nan)
netflix_imdb['duration_min_movie'] = netflix_imdb['duration_min_movie'].str.replace('[A-Za-z]+', '').astype(float)
# Creating the dataset with the mean IMDb score for each type.
netflix_imdb_mean_scores = netflix_imdb.groupby('type_new').agg({'imdb_score':'mean'}).sort_values('imdb_score', ascending=False)
netflix_imdb_mean_scores.index = ['TV shows', 'Cinema movies', 'TV movies']
# Adding the total mean for all types.
netflix_imdb_mean_scores = pd.concat([pd.DataFrame([netflix_imdb['imdb_score'].mean()], index=['All'], columns=['imdb_score']), netflix_imdb_mean_scores])

# Setting the style of the graph.
sns.set_style('white')
# Setting the resoulution.
plt.figure(dpi=100)
# Generating the actual bar chart.
plt.bar(x=netflix_imdb_mean_scores.index, height=netflix_imdb_mean_scores['imdb_score'], color=['black', '#E50914'])
# Adjusting the y axis.
plt.ylim(6,7.4)
plt.ylabel('IMDb score')

# Adding the mean values on top of the bars.
for i, count in enumerate(netflix_imdb_mean_scores['imdb_score']):
    plt.text(i, count+.025, round(count, 2), ha='center', fontweight='bold')

plt.title('Netflix US catalogue in 2019\nAverage IMDb score per production type', fontsize=15)

plt.show()
# Creating the scatter plot for the IMDb scores and the duration.
g = sns.lmplot('release_year', 'imdb_score', data=netflix_imdb[netflix_imdb['type'] == 'Movie'], markers='1', hue='type', palette=['#E50914'], fit_reg=False)
# Adjusting the labels, legend and resolution of the plot.
g.set(ylim=(1, 10), xlabel='Release year', ylabel='IMDb score')
g._legend.set_title('')
g.fig.set_dpi(100)

plt.title('Netflix US catalogue in 2019\nRelease year and IMDb score', fontsize=15)
plt.show()

# Printing the correlation between the variables (Pearson because the variables are both continuous).
correlation = netflix_imdb.loc[netflix_imdb['type'] == 'Movie', 'imdb_score'].corr(netflix_imdb.loc[netflix_imdb['type'] == 'Movie', 'release_year'], method='pearson')
print('Correlation (Pearson): {:.2}'.format(correlation))
# Creating the scatter plot for the IMDb scores and the release year.
g = sns.lmplot('duration_min_movie', 'imdb_score', data=netflix_imdb[netflix_imdb['type'] == 'Movie'], markers=['1'], hue='type', palette=['#E50914'], fit_reg=False)
g.set(ylim=(1, 10), xlabel='Duration (in minutes)', ylabel='IMDb score')
g._legend.set_title('')
g.fig.set_dpi(100)

plt.title('Netflix US catalogue in 2019\nMovie length and IMDb score', fontsize=15)
plt.show()

correlation = netflix_imdb.loc[netflix_imdb['type'] == 'Movie', 'imdb_score'].corr(netflix_imdb.loc[netflix_imdb['type'] == 'Movie', 'duration_min_movie'], method='pearson')
print('Correlation (Pearson): {:.2}'.format(correlation))
# Creating the scatter plot for the IMDb scores and the rating number.
production_types = ['TV show', 'Cinema movie', 'TV movie']
# We additionally jitter the values along the x axis to aid the visualisation (otherwise there would be large gaps between the values).
g = sns.lmplot('rating_number', 'imdb_score', hue='type_new', hue_order=production_types, data=netflix_imdb, x_jitter=.5, markers=['1', '2', '3'], palette=['#E50914', 'black', '#A9A9A9'], fit_reg=False)
g.set(xlim=(1, 6), ylim=(1, 10), xlabel='Age rating (jittered)', ylabel='IMDb score')
g._legend.set_title('')
g.fig.set_dpi(100)

plt.title('Netflix US catalogue in 2019\nAge ratings and IMDb score', fontsize=15)
plt.show()

# Printing the correlations between the variables (Spearman because the rating_number variable actually has ordered categories).
correlations = []
for each_type in production_types:
    correlation = netflix_imdb.loc[netflix_imdb['type_new'] == each_type, 'imdb_score'].corr(netflix_imdb.loc[netflix_imdb['type_new'] == each_type, 'rating_number'], method='spearman')
    correlations.append(correlation)

print('Correlations (Spearman)')
for i in range(3):
    print('{}: {:.2f}'.format(production_types[i], correlations[i]))