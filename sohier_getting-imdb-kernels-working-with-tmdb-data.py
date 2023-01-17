import json

import pandas as pd
def load_tmdb_movies(path):

    df = pd.read_csv(path)

    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())

    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df





def load_tmdb_credits(path):

    df = pd.read_csv(path)

    json_columns = ['cast', 'crew']

    for column in json_columns:

        df[column] = df[column].apply(json.loads)

    return df
# Columns that existed in the IMDB version of the dataset and are gone.

LOST_COLUMNS = [

    'actor_1_facebook_likes',

    'actor_2_facebook_likes',

    'actor_3_facebook_likes',

    'aspect_ratio',

    'cast_total_facebook_likes',

    'color',

    'content_rating',

    'director_facebook_likes',

    'facenumber_in_poster',

    'movie_facebook_likes',

    'movie_imdb_link',

    'num_critic_for_reviews',

    'num_user_for_reviews'

                ]



# Columns in TMDb that had direct equivalents in the IMDB version. 

# These columns can be used with old kernels just by changing the names

TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {

    'budget': 'budget',

    'genres': 'genres',

    'revenue': 'gross',

    'title': 'movie_title',

    'runtime': 'duration',

    'original_language': 'language',  # it's possible that spoken_languages would be a better match

    'keywords': 'plot_keywords',

    'vote_count': 'num_voted_users',

                                         }



IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}





def safe_access(container, index_values):

    # return a missing value rather than an error upon indexing/key failure

    result = container

    try:

        for idx in index_values:

            result = result[idx]

        return result

    except IndexError or KeyError:

        return pd.np.nan





def get_director(crew_data):

    directors = [x['name'] for x in crew_data if x['job'] == 'Director']

    return safe_access(directors, [0])





def pipe_flatten_names(keywords):

    return '|'.join([x['name'] for x in keywords])





def convert_to_original_format(movies, credits):

    # Converts TMDb data to make it as compatible as possible with kernels built on the original version of the data.

    tmdb_movies = movies.copy()

    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)

    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)

    # I'm assuming that the first production country is equivalent, but have not been able to validate this

    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['director_name'] = credits['crew'].apply(get_director)

    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [0, 'name']))

    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))

    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))

    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)

    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)

    return tmdb_movies
movies = load_tmdb_movies("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")

credits = load_tmdb_credits("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

original_format =convert_to_original_format(movies, credits)
with open("../input/static-copy-of-recommendation-engine-notebook/recommendation_engine.ipynb", "r") as f_open:

    raw_notebook = f_open.readlines()

for line in raw_notebook[:10]:

    print(line)
with open("../input/static-copy-of-recommendation-engine-notebook/recommendation_engine.ipynb", "r") as f_open:

    df = pd.DataFrame(json.load(f_open)['cells'])
df.head(3)
import re





def rows_with_lost_columns(code_lines): 

    lost_column_pattern = '\W(' + '|'.join(LOST_COLUMNS) + ')\W'

    # adding one  to the output since text editors usually use one based indexing.

    troubled_lines = [line_number + 1 for line_number, line in enumerate(code_lines)

                      if bool(re.search(lost_column_pattern, line))]

    if troubled_lines:

        return troubled_lines

    return None
df['lost_column_lines'] = df.apply(lambda row: rows_with_lost_columns(row['source'])

                                   if row['cell_type'] == 'code' else pd.np.nan, axis=1)
num_lines_to_review = df.lost_column_lines.apply(lambda x: len(x) if type(x) == list else 0).sum()

print(f'{num_lines_to_review} lines of code need to be reviewed')
df[['source', 'lost_column_lines']][~df.lost_column_lines.isnull()]