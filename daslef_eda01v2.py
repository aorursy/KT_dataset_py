import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

 

import os

print(os.listdir("../input"))
raw = pd.read_csv('../input/train.csv')
raw.head(3)
raw.columns
raw.info()
raw.describe()
raw['belongs_to_collection'].value_counts()
cols_to_eval = ['belongs_to_collection', 'genres', 'production_companies','production_countries', 

                'spoken_languages', 'Keywords', 'cast', 'crew']



for i in cols_to_eval:

    raw[i] = raw[i].apply(lambda x: eval(x) if isinstance(x, str) else x)
raw.head()
raw['has_collection'] = raw['belongs_to_collection'].apply(lambda x: 1 if isinstance(x, list) else 0)
raw.has_collection.value_counts()
raw[pd.notnull(raw['belongs_to_collection'])]['belongs_to_collection']

collection_counter = (('James Bond Collection', 16), ('Friday the 13th Collection', 7), 

('The Pink Panther (Original) Collection', 6), ('Police Academy Collection',5), ('Pokémon Collection',5),        

('Alien Collection',5), ('Transformers Collection', 4), ('Rambo Collection', 4), ('Ice Age Collection',4),               

('Paranormal Activity Collection',4), ('Resident Evil Collection', 4), ('Rocky Collection', 4), ("Child's Play Collection", 4),   

('The Fast and the Furious Collection', 4))
collection_names = raw.belongs_to_collection.apply(lambda x: x[0]['name'] if pd.notnull(x) else x).value_counts()[:116].index.values
for i in collection_names:

    raw[i] = raw.belongs_to_collection.apply(lambda x: 1 if pd.notnull(x) and x[0]['name'] == i else 0)
raw.drop(['belongs_to_collection'], axis=1, inplace=True)
raw.head(1)
raw[raw['budget'] == 0]['genres'].apply(lambda x: x[0]['name'] if isinstance(x, 

                                list) else x).value_counts()
[print(i) for i in raw[raw['budget'] == 0]['genres'].values]
raw[raw['budget'] != 0].describe()
raw.loc[raw['budget'] == 0,'budget'] = 3.089305e+07

raw.head()
raw['genres'] = raw['genres'].apply(lambda x: [i['name'] for i in x] 

                                            if isinstance(x,list) else [])

raw['genres_number'] = raw['genres'].apply(lambda x: len(x))

raw.head()
top_genres = [k for k,v in [('Drama', 1531), ('Comedy', 1028), ('Thriller', 789), ('Action', 741), ('Romance', 571), ('Crime', 469), ('Adventure', 439), ('Horror', 301), ('Science Fiction', 290), ('Family', 260), ('Fantasy', 232), ('Mystery', 225), ('Animation', 141), ('History', 132), ('Music', 

    100), ('War', 100), ('Documentary', 87), ('Western', 43)]]
top_genres
for i in top_genres:

    raw['genre_' + i] = raw['genres'].apply(lambda x: 1 if i in x else 0)
raw.drop(['genres'], axis=1, inplace=True)
raw[pd.notnull(raw['homepage'])]['homepage'].value_counts()
raw['has_homepage'] = raw['homepage'].apply(lambda x: 1 if pd.notnull(x) else 0)

raw.drop(['homepage'],axis=1,inplace=True)
raw.drop(['original_title'],axis=1,inplace=True)
raw.drop(['status'],axis=1,inplace=True)
raw.head()
raw['release_date'] = raw['release_date'].apply(lambda x: x.split('/'))
raw['release_date'].head()
raw['release_year'] = raw['release_date'].apply(lambda x: '19' + x[2] if int(x[2]) >= 20 else '20' + x[2]).astype(int)
raw['release_day'] = raw['release_date'].apply(lambda x: x[1]).astype(int)
raw['release_month'] = raw['release_date'].apply(lambda x: x[0]).astype(int)
raw['release_day_of_year'] = raw['release_day'] + (raw['release_month'] - 1) * 30
raw.head()
raw.drop(['release_date'], axis=1, inplace=True)
raw['production_companies'] = raw['production_companies'].apply(lambda x: [i['name'] for i in x] 

                                            if isinstance(x,list) else [])
raw['production_companies_number'] = raw['production_companies'].apply(lambda x: len(x))

raw.head()
raw['production_companies_number'].value_counts().plot.bar()
all_companies = []
for i in raw['production_companies'].values:

    for j in i:

        all_companies.append(j)
# all_companies = set(all_companies)
top_companies = {company: all_companies.count(company) for company in set(all_companies)}

top_companies.items()
top_companies = sorted(top_companies.items(), key=lambda x: x[1], reverse=True)
pd.Series([i[1] for i in top_companies]).plot.line()
top_companies = [i[0] for i in top_companies if i[1] >= 5]
raw['production_companies'][2]
for i in top_companies:

    raw['company_' + i] = raw['production_companies'].apply(lambda x: 1 if i in x else 0)
# list(filter(lambda x: x[1] >= 5, top_companies))
raw.head()
raw['production_countries'].apply(lambda x: len(x) if isinstance(x, list) else 0).value_counts().plot.bar()
from collections import Counter

raw['production_countries'] = raw['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

top_countries = [i[0] for i in Counter([i for j in list_of_countries for i in j]).most_common(25)]
raw['countries_number'] = raw['production_countries'].apply(lambda x: len(x) if isinstance(x, list) else 0)



for g in top_countries:

    raw['production_country_' + g] = raw['production_countries'].apply(lambda x: 1 if g in x else 0)
raw.drop(['production_countries'], axis=1, inplace=True)
raw.head()