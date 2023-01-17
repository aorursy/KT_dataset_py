# import modules

import pandas as pd

import numpy as np

import pycountry

import re

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer



# load data

data = pd.read_csv('../input/un-general-debates.csv')
# peak into data

data.head()
# add column for full country name

country_name = []

for code in data['country']:

    try:

        country_name.append(pycountry.countries.lookup(code).name)

    except LookupError:

        country_name.append('')

data['country_name'] = country_name

        

# convert text data to lower case (for easier analysis)

data['text'] = data['text'].str.lower()
# Remove unusual symbols from description

def clean(s):    

    # Remove any tags:

    cleaned = re.sub(r"(?s)<.?>", " ", s)

    # Keep only regular chars:

    cleaned = re.sub(r"[^A-Za-z0-9(),*!?\'\`]", " ", cleaned)

    # Remove unicode chars

    cleaned = re.sub("\\\\u(.){4}", " ", cleaned)

    return cleaned.strip()



# clean text

data['text'] = data.text.apply(lambda x: clean(x))
# remove data with null value in year column

data = data[data['year'].notnull()]



# drop session column -- provides no information

data = data.drop(['session'], axis=1)



# Group data by country and into 5 year periods

# e.g. USA (2000-2004), China (1980-1984)

data['year'] = (data['year'] / 5).astype(int)*5

data = data.groupby(['country', 'year', 'country_name'])['text'].apply(list)

data = data.apply(lambda x: ''.join(x))

data = data.reset_index(drop=False)



data[:20]
# Create 5000 TF-IDF features, using 3-gram

num_features = 5000

tfidf = TfidfVectorizer(max_features = num_features, strip_accents='unicode',

                        lowercase=True, stop_words='english', ngram_range=(1,3))

print('Fitting Data...')

tfidf.fit(data['text'].values.astype('U'))



print('Starting Transform...')

text_tfidf = tfidf.transform(data['text'])



print('Label and Incorporate TF-IDF...')

data_array = pd.DataFrame(text_tfidf.toarray())

feature_names = tfidf.get_feature_names()



for i in range(num_features):

    feature_names[i] = 'TF_' + feature_names[i]



data_array.columns = feature_names

data = pd.concat([data, data_array], axis=1)



data[:2]
# create list of TF-IDF features, confirm they make sense

features = data.columns.tolist()

for i in ['year', 'country', 'country_name', 'text']:

    features.remove(i)

features
def word_similiarity(country, year):

    '''finds similiarity of words used by different countries'''

    df = data.copy()

    primary = data[((data['country_name'] == country) | (data['country'] == country)) & (data['year']==year)]

    for i in features:

        df[i] = [np.square(primary[i].values[0].astype(float) - x) for x in data[i]]



    df['total'] = df[features].sum(axis=1).abs()

    df = df.sort_values(by='total', ascending=True).reset_index(drop=True)

    return df
def closest_countries(country, year, count=10, same_year=True):

    '''finds most similiar and least similiar countries'''

    df = word_similiarity(country, year)

    if same_year == True:

        df = df[df['year'] == year]

        most_similiar = df[1:1 + count].country_name

        least_similiar = df[-count:].country_name

    else:

        most_similiar = df[1:1 + count].country_name + [' ' for i in range(count)] + ['(' + str(i) + ')' for i in df[1:1 + count].year]

        least_similiar = df[-count:].country_name + [' ' for i in range(count)] + ['(' + str(i) + ')' for i in df[-count:].year]

    print(country + ', ' + str(year))

    print('most similiar:')

    print(most_similiar.values)

    print()

    print('least similiar:')

    print(least_similiar.values)

    print()

    print()

    return ''
# Let's take a look at which countries the

# USA shares the same language with throughout time

closest_countries('USA', 1980)

closest_countries('USA', 1990)

closest_countries('USA', 2000)

closest_countries('USA', 2010)

closest_countries('USA', 2010, 10, False)
# Just for fun, here are other countries:

closest_countries('China', 2015)

closest_countries('India', 2015)

closest_countries('United Kingdom', 2015)

closest_countries('France', 2015)

closest_countries('Canada', 2015)

closest_countries('Israel', 2015)

closest_countries('Central African Republic', 2015)