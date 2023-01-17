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
movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")

credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")
movies.head(3)
credits.head(3)
print(sorted(credits.cast.iloc[0][0].keys()))
print(sorted(credits.crew.iloc[0][0].keys()))
[actor['name'] for actor in credits['cast'].iloc[0][:5]]
def safe_access(container, index_values):

    # return a missing value rather than an error upon indexing/key failure

    result = container

    try:

        for idx in index_values:

            result = result[idx]

        return result

    except IndexError or KeyError:

        return pd.np.nan
credits['gender_of_lead'] = credits.cast.apply(lambda x: safe_access(x, [0, 'gender']))

credits['lead'] = credits.cast.apply(lambda x: safe_access(x, [0, 'name']))

credits.head(3)
credits.gender_of_lead.value_counts()
df = pd.merge(movies, credits, left_on='id', right_on='movie_id')

df[['original_title', 'revenue', 'lead', 'gender_of_lead']].sort_values(by=['revenue'], ascending=False)[:10]
credits.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['cast']], axis=1);

credits.apply(lambda row: [x.update({'movie_id': row['movie_id']}) for x in row['crew']], axis=1);

credits.apply(lambda row: [person.update({'order': order}) for order, person in enumerate(row['crew'])], axis=1);



cast = []

credits.cast.apply(lambda x: cast.extend(x))

cast = pd.DataFrame(cast)

cast['type'] = 'cast'



crew = []

credits.crew.apply(lambda x: crew.extend(x))

crew = pd.DataFrame(crew)

crew['type'] = 'crew'



people = pd.concat([cast, crew],  ignore_index=True)
people.sample(3)