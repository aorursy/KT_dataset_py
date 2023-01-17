# imports

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import ast



import plotly.graph_objs as go 

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True) 
movie_meta = pd.read_csv('../input/movies_metadata.csv', low_memory = False)

movie_meta.head()
movie_meta.dtypes
movie_meta.shape
movie_meta = movie_meta.set_index('title', drop = True)
sns.heatmap(movie_meta.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
movie_meta = movie_meta.dropna(thresh = 5)
movie_meta = movie_meta.drop(['homepage', 'imdb_id', 'original_title', 'overview', 'popularity', 'poster_path', 'status', 'tagline', 'video'],

               axis = 1)
movie_meta['adult'].unique()
movie_meta.loc[(movie_meta['adult'] != 'True') & (movie_meta['adult'] != 'False'), 'adult'] = False
movie_meta['adult'] = movie_meta['adult'].map({'True': True, 'False': False})
movie_meta['belongs_to_collection'] = movie_meta['belongs_to_collection'].notna()
movie_meta[movie_meta['budget'] == '0']['budget'].count()
movie_meta[movie_meta['revenue'] == 0]['revenue'].count()
movie_meta['budget'] = pd.to_numeric(movie_meta['budget'], errors = 'coerce')

# revenue is already a float

movie_meta.loc[(movie_meta['budget'] == 0) & (movie_meta['revenue'] == 0), 'revenue'] = np.nan

movie_meta.loc[movie_meta['budget'] == 0, 'budget'] = np.nan
movie_meta[movie_meta['budget'] < 100]['budget'].count()
def scale(num):

    if num < 100:

        return num * 1000000

    elif num >= 100 and num < 1000:

        return num * 1000

    else:

        return num
movie_meta[['budget', 'revenue']] = movie_meta[['budget', 'revenue']].applymap(scale)
sns.distplot(movie_meta[movie_meta['budget'].notnull()]['budget'])
def get_values(data_str):

    if isinstance(data_str, float):

        pass

    else:

        values = []

        data_str = ast.literal_eval(data_str)

        if isinstance(data_str, list):

            for k_v in data_str:

                values.append(k_v['name'])

            return values

        else:

            return None
movie_meta[['genres', 'production_companies', 'production_countries', 'spoken_languages']] = movie_meta[['genres', 'production_companies', 'production_countries', 'spoken_languages']].applymap(get_values)
movie_meta['release_date'] = pd.to_datetime(movie_meta['release_date'], format = '%Y-%m-%d', errors='coerce')
movie_meta.head()
sns.heatmap(movie_meta.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
movie_meta['profit'] = movie_meta['revenue'] - movie_meta['budget']
sns.heatmap(movie_meta.corr(), cmap = 'YlGnBu')
def counting_values(df, column):

    value_count = {}

    for row in df[column].dropna():

        if len(row) > 0:

            for key in row:

                if key in value_count:

                    value_count[key] += 1

                else:

                    value_count[key] = 1

        else:

            pass

    return value_count
countries = pd.Series(counting_values(movie_meta, 'production_countries'))



#the map didn't come out so interesting in a regular scale, so I changed it to a logarithmic one:

ln_countries = pd.Series(np.log(countries.values), index = countries.index)
data = dict(type = 'choropleth',

           locations = ln_countries.index,

           locationmode = 'country names',

           colorscale = 'Blackbody',

           text = countries.values,

           z = ln_countries.values,

           colorbar = {'title': 'log of Sum of Movies'})



layout = dict(title = 'Movies Filmed in Countries around the World',

             geo = dict(showframe = False,

                       projection = {'type': 'Natural Earth'}))



choromap = go.Figure(data = [data],layout = layout)

iplot(choromap,validate=False)
genres_count = pd.Series(counting_values(movie_meta, 'genres'))

genres_count.sort_values(ascending = False).head(10).plot(kind = 'bar')
genres_count = pd.Series(counting_values(movie_meta[movie_meta['belongs_to_collection'] == True], 'genres'))

genres_count.sort_values(ascending = False).head(10).plot(kind = 'bar')
movie_meta['years'] = movie_meta['release_date'].apply(lambda x: x.year)



movie_meta[(movie_meta['years'] < 2018) & (movie_meta['years'] >= 1950)].groupby(by = 'years').mean()['vote_average'].plot()