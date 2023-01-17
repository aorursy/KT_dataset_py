import re

import gc

gc.enable()



import numpy as np

import pandas as pd

from collections import defaultdict



import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats



%matplotlib inline



import warnings  

warnings.filterwarnings('ignore')
# Function to reduce memory usage.  From: https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape)

train.sample(2)
# Revenues are not uniformally distributed

train['revenue'].hist(bins=25)

plt.show()
# When comparing the listed revenues with their actual values found online, 

# it's clear the values given here are not accurate.

train.sort_values('revenue').sample(3)
# Budget is also skewed.

train['budget'].hist(bins=25)

plt.show()
# $0 budget for some movies?

train['budget'].describe()
print('Movies with 0$ Budget:', len(train[train['budget'] == 0]))

train[train['budget'] == 0].head()
# Create columns for year, month, and day of week

train['release_date'] = pd.to_datetime(train['release_date'], infer_datetime_format=True)

train['release_day'] = train['release_date'].apply(lambda t: t.day)

train['release_weekday'] = train['release_date'].apply(lambda t: t.weekday())

train['release_month'] = train['release_date'].apply(lambda t: t.month)



# Year was being interpreted as future dates in some cases so I had to adjust some values

train['release_year'] = train['release_date'].apply(lambda t: t.year if t.year < 2018 else t.year -100)
train['runtime'].hist(bins=25)

plt.show()
len(train[train['runtime'] == 0])
# I'll write a function that will map the average runtime for each year to movies with 0 runtie.

from collections import defaultdict

def map_runtime(df):

    df['runtime'].fillna(0)

    

    run = df[(df['runtime'].notnull()) & (df['runtime'] != 0)]

    year_mean = run.groupby(['release_year'])['runtime'].agg('mean')

    d = dict(year_mean)

    

    for i in df[df['runtime'] == 0]:

        df['runtime'] = df.loc[:, 'release_year'].map(d)

    

    return df
train = map_runtime(train)

train.runtime.describe()
train['homepage'].head()
# For homepage, I'll change it to 0 for NaN and 1 if a page is listed.

train['homepage'].fillna(0, inplace=True)

train.loc[train['homepage'] != 0, 'homepage'] = 1
# For poster_path, I'll change it to 0 for NaN and 1 if a path is listed.

train['poster_path'].fillna(0, inplace=True)

train.loc[train['poster_path'] != 0, 'poster_path'] = 1
# For genres, I'll fill Na values with drama (most common).  Likely a better approach available.

train.genres = train.genres.fillna('18')
# To fill in zero budget data points, I'll try to use correlated values as predictors

X = train[train['budget'] != 0]

for i in X.select_dtypes(include='number', exclude='datetime'):

    print(i, stats.pearsonr(X.budget, X[i]))
# release_year and popularity correlate most strongly with budget

def map_budget(df):

    d = defaultdict()

    #df['budget'] = df['budget'].fillna(0)

    X = df[df['budget'] != 0]

    

    year_mean = pd.Series(X.groupby(['release_year'])['budget'].agg('mean'))

    d = dict(year_mean)

    

    for i in df[df['budget'] == 0]:

        df['budget'] = df.loc[:, 'release_year'].map(d)

    

    # In a few cases, there are only 1 or 2 movies provided from a given year and are filled with Na values

    df.budget = df.sort_values(by='release_year').budget.fillna(method='ffill')

    

    return df
train = map_budget(train)

train.budget.describe()
train['belongs_to_collection'].head()
# belongs_to_collection NaN values can be replaced with 'none'

train['belongs_to_collection'] = train['belongs_to_collection'].fillna('none')
train['spoken_languages'].head()
train.spoken_languages.value_counts(dropna=False)
# For spoken_languages I'll fill Na values with [{'iso_639_1': 'en', 'name': 'English'}]

train.spoken_languages = train.spoken_languages.fillna("[{'iso_639_1': 'en', 'name': 'English'}]")
train['overview'].head()
# For overview, I'll fill Na values with 'none'

train.overview = train.overview.fillna('none')
train['Keywords'].head()
# For Keywords, I'll fill Na values with 'none'

train.Keywords = train.Keywords.fillna('none')
train.production_countries.describe()
# For production_countries, I'll fill Na with the most common value

train.production_countries = train.production_countries.fillna("[{'iso_3166_1': 'US', 'name': 'United States of America'}]")
train.production_companies.value_counts()
# Create a columns for title length

title_len = []

for i in train['title']:

    title_len.append(len(i.split()))

title_len = pd.Series(title_len, name='title_length')

train = pd.concat([train,title_len], axis=1)



train['title_length'].describe()
# For genres, I'll make a new column counting the number of listed genre types

# This will strip out all characters except for numbers, and return this as an array

genre_ids = []

for i in train['genres']:

    i = re.findall('\d+', i)

    genre_ids.append(i)

genre_ids = pd.Series(genre_ids, name='genre_ids').astype(str)



# This will count the number of genres listed for each film

num_genre_types = []

for i in genre_ids:

    num_genre_types.append(len(i.split()))

num_genre_types = pd.Series(num_genre_types, name='num_genre_types').astype(int)

train = pd.concat([train, genre_ids, num_genre_types], axis=1)



train['num_genre_types'].describe()
# Create column for sequels

is_sequel = []

for i in train['Keywords']:

    if 'sequel' in str(i):

        is_sequel.append(1)

    else:

        is_sequel.append(0)

is_sequel = pd.Series(is_sequel, name='is_sequel')

train = pd.concat([train, is_sequel], axis=1)



train['is_sequel'].describe()
keyword_words = []

for i in train['Keywords']:

    i = re.findall('[a-zA-Z \t]+', i)

    stopwords = ['id', 'name', ' ']

    i = [word for word in i if word not in stopwords]

    keyword_words.append(i)

keyword_words = pd.Series(keyword_words, name='keyword_words').astype(str)

train = pd.concat([train, keyword_words], axis=1)



# This will count the number of Keywords listed for each film

num_keywords = []

for i in keyword_words:

    num_keywords.append(len(i.split(',')))

num_keywords = pd.Series(num_keywords, name='num_keywords').astype(int)

train = pd.concat([train, num_keywords], axis=1)



train['num_keywords'].describe()
# could use the numbers from the categories, sum them up, and then convert them to a category to target incode

keyword_ids = []

for i in train['Keywords']:

    i = re.findall('[0-9]+', i)

    keyword_ids.append(i)

keyword_ids = pd.Series(keyword_ids, name='keyword_ids')

train = pd.concat([keyword_ids, train], axis=1)

train.keyword_ids.head()
train.belongs_to_collection.head()
# Extract number from belongs to collection

collection_id = []

for i in train['belongs_to_collection']:

    i = re.findall('[0-9]+', i)

    collection_id.append(i[:1])

collection_id = pd.Series(collection_id, name='collection_id').apply(lambda x: ''.join([str(i) for i in x]))



# Fill in blank values with 'No Collection'

for i in collection_id[collection_id == ''].index:

    collection_id.loc[i] = 'No Collection'



train = pd.concat([train, collection_id], axis=1)



train['collection_id'].describe()
# Add column with 1 for movies in a collection and 0 if not

is_in_collection = []

for i in train['collection_id']:

    if i != 'No Collection':

        is_in_collection.append(1)

    else:

        is_in_collection.append(0)



is_in_collection = pd.Series(is_in_collection, name='is_in_collection')

train = pd.concat([train, is_in_collection], axis=1)



train['is_in_collection'].describe()
train['production_countries'].head()
# Create a column for production country (1 for US, 0 for rest of world)

# It would be helpful if countries had different codes, but they all appear to be the same so it's difficult to work with

US_prod_country = []

for i in train['production_countries']:

    if 'US' in str(i):

        US_prod_country.append(1)

    else:

        US_prod_country.append(0)

US_prod_country = pd.Series(US_prod_country, name='US_prod_country')

train = pd.concat([train, US_prod_country], axis=1)



train['US_prod_country'].describe()
# Create column for number of production countries

num_production_countries = []

for i in train['production_countries']:

    i = re.findall('[a-zA-Z \t]+', str(i))

    num_production_countries.append(str(i).count('name'))

num_production_countries = pd.Series(num_production_countries, name='num_production_countries')

train = pd.concat([train, num_production_countries], axis=1)



train['num_production_countries'].describe()
# Create a column for each production company name and a column for the number of companies

production_company_names = []

num_production_companies = []

for i in train['production_companies']:

    i = re.findall('[a-zA-Z \t]+', str(i))

    stopwords = ['id', 'name', ' ']

    production_company_names.append([word for word in i if word not in stopwords])

    num_production_companies.append(str(i).count('name'))



production_company_1 = []

production_company_2 = []

production_company_3 = []

production_company_4 = []

production_company_5 = []

production_company_6 = []

production_company_7 = []

production_company_8 = []



for i in production_company_names:

    try:

        production_company_1.append(i[:][0:1])

        production_company_2.append(i[:][1:2])

        production_company_3.append(i[:][2:3])

        production_company_4.append(i[:][3:4])

        production_company_5.append(i[:][4:5])

        production_company_6.append(i[:][5:6])

        production_company_7.append(i[:][6:7])

        production_company_8.append(i[:][7:8])

    except:

        production_company_1.append('none')

        production_company_2.append('none')

        production_company_3.append('none')

        production_company_4.append('none')

        production_company_5.append('none')

        production_company_6.append('none')

        production_company_7.append('none')

        production_company_8.append('none')



num_production_companies = pd.Series(num_production_companies, name='num_production_companies')

production_company_1 = pd.Series(production_company_1, name='production_company_1').apply(''.join)

for i in production_company_1[production_company_1 == ''].index:

    production_company_1.iloc[i] = False

production_company_2 = pd.Series(production_company_2, name='production_company_2').apply(''.join)

for i in production_company_2[production_company_2 == ''].index:

    production_company_2.iloc[i] = False

production_company_3 = pd.Series(production_company_3, name='production_company_3').apply(''.join)

for i in production_company_3[production_company_3 == ''].index:

    production_company_3.iloc[i] = False

production_company_4 = pd.Series(production_company_4, name='production_company_4').apply(''.join)

for i in production_company_4[production_company_4 == ''].index:

    production_company_4.iloc[i] = False

production_company_5 = pd.Series(production_company_5, name='production_company_5').apply(''.join)

for i in production_company_5[production_company_5 == ''].index:

    production_company_5.iloc[i] = False

production_company_6 = pd.Series(production_company_6, name='production_company_6').apply(''.join)

for i in production_company_6[production_company_6 == ''].index:

    production_company_6.iloc[i] = False

production_company_7 = pd.Series(production_company_7, name='production_company_7').apply(''.join)

for i in production_company_7[production_company_7 == ''].index:

    production_company_7.iloc[i] = False

production_company_8 = pd.Series(production_company_8, name='production_company_8').apply(''.join)

for i in production_company_8[production_company_8 == ''].index:

    production_company_8.iloc[i] = False

train = pd.concat([train, num_production_companies, production_company_1, production_company_2,

              production_company_3, production_company_4, production_company_5, production_company_6,

              production_company_7, production_company_8], axis=1)



train.production_company_8.head()
# Create a column for number of spoken languages

num_spoken_languages = []

for i in train['spoken_languages']:

    a = str(i).split()

    num_spoken_languages.append(a.count("'name':"))

num_spoken_languages = pd.Series(num_spoken_languages, name = 'num_spoken_languages')

train = pd.concat([train, num_spoken_languages], axis=1)



train['num_spoken_languages'].describe()
# Create column for release status

status_is_released = []

for i in train['status']:

    if i == 'Released':

        status_is_released.append(1)

    else:

        status_is_released.append(0)

status_is_released = pd.Series(status_is_released, name = 'status_is_released')

train = pd.concat([train, status_is_released], axis=1)

train['status_is_released'].describe()
def data_processing(df):

    # Create columns for year, month, and day of week

    df['release_date'] = df['release_date'].fillna(method='ffill')

    df['release_date'] = pd.to_datetime(df['release_date'], infer_datetime_format=True)

    df['release_day'] = df['release_date'].apply(lambda t: t.day)

    df['release_weekday'] = df['release_date'].apply(lambda t: t.weekday())

    df['release_month'] = df['release_date'].apply(lambda t: t.month)

    # Year was being interpreted as future dates in some cases so I had to adjust some values

    df['release_year'] = df['release_date'].apply(lambda t: t.year if t.year < 2018 else t.year -100)

    

    # Function that will map the average runtime for each year to movies with 0 runtie.

    def map_runtime(df):

        df['runtime'].fillna(0)

    

        run = df[(df['runtime'].notnull()) & (df['runtime'] != 0)]

        year_mean = run.groupby(['release_year'])['runtime'].agg('mean')

        d = dict(year_mean)

    

        for i in df[df['runtime'] == 0]:

            df['runtime'] = df.loc[:, 'release_year'].map(d)

        return df

    df = map_runtime(df)

    

    # For homepage, I'll change it to 0 for NaN and 1 if a page is listed.

    df['homepage'].fillna(0, inplace=True)

    df.loc[df['homepage'] != 0, 'homepage'] = 1

    

    # For poster_path, I'll change it to 0 for NaN and 1 if a path is listed.

    df['poster_path'].fillna(0, inplace=True)

    df.loc[df['poster_path'] != 0, 'poster_path'] = 1

    

    # release_year correlates strongly with budget, so I'll use that to estimate the null values

    def map_budget(df):

        d = defaultdict()

        X = df[df['budget'] != 0]

        year_mean = pd.Series(X.groupby(['release_year'])['budget'].agg('mean'))

        d = dict(year_mean)

    

        for i in df[df['budget'] == 0]:

            df['budget'] = df.loc[:, 'release_year'].map(d)

    

        # In a few cases, there are only 1 or 2 movies provided from a given year and are filled with Na values

        df.budget = df.sort_values(by='release_year').budget.fillna(method='ffill')

        return df

    df = map_budget(df)

    

    # Fill remaining Na values

    df['belongs_to_collection'] = df['belongs_to_collection'].fillna('none')

    df.spoken_languages = df.spoken_languages.fillna("[{'iso_639_1': 'en', 'name': 'English'}]")

    df.overview = df.overview.fillna('none')

    df.Keywords = df.Keywords.fillna('none')

    df.production_countries = df.production_countries.fillna(

        "[{'iso_3166_1': 'US', 'name': 'United States of America'}]")

    df.genres = df.genres.fillna('18')

    

    ############ Feature Engineering ############

    

    # Create a columns for title length

    title_len = []

    for i in df['title']:

        title_len.append(len(str(i).split()))

    title_len = pd.Series(title_len, name='title_length')

    df = pd.concat([df, title_len], axis=1)

    

    # Create columns for genres id's and for number of genres listed

    genre_id = []

    num_genre_types = []

    for i in df['genres']:

        i = re.findall('\d+', str(i))

        genre_id.append(i)

    genre_id = pd.Series(genre_id, name='genre_id') #.apply(lambda x: ''.join([str(i) for i in x]))

    

    genre_id_1 = []

    genre_id_2 = []

    genre_id_3 = []

    genre_id_4 = []

    genre_id_5 = []

    genre_id_6 = []

    genre_id_7 = []



    for i in genre_id:

        try:

            genre_id_1.append(i[:][0:1])

            genre_id_2.append(i[:][1:2])

            genre_id_3.append(i[:][2:3])

            genre_id_4.append(i[:][3:4])

            genre_id_5.append(i[:][4:5])

            genre_id_6.append(i[:][5:6])

            genre_id_7.append(i[:][6:7])

        except:

            genre_id_1.append('none')

            genre_id_2.append('none')

            genre_id_3.append('none')

            genre_id_4.append('none')

            genre_id_5.append('none')

            genre_id_6.append('none')

            genre_id_7.append('none')

            

    genre_id_1 = pd.Series(genre_id_1, name='genre_id_1').apply(''.join)

    for i in genre_id_1[genre_id_1 == ''].index:

        genre_id_1.iloc[i] = 'none'

    genre_id_2 = pd.Series(genre_id_2, name='genre_id_2').apply(''.join)

    for i in genre_id_2[genre_id_2 == ''].index:

        genre_id_2.iloc[i] = 'none'

    genre_id_3 = pd.Series(genre_id_3, name='genre_id_3').apply(''.join)

    for i in genre_id_3[genre_id_3 == ''].index:

        genre_id_3.iloc[i] = 'none'

    genre_id_4 = pd.Series(genre_id_4, name='genre_id_4').apply(''.join)

    for i in genre_id_4[genre_id_4 == ''].index:

        genre_id_4.iloc[i] = 'none'

    genre_id_5 = pd.Series(genre_id_5, name='genre_id_5').apply(''.join)

    for i in genre_id_5[genre_id_5 == ''].index:

        genre_id_5.iloc[i] = 'none'

    genre_id_6 = pd.Series(genre_id_6, name='genre_id_6').apply(''.join)

    for i in genre_id_6[genre_id_6 == ''].index:

        genre_id_6.iloc[i] = 'none'

    genre_id_7 = pd.Series(genre_id_7, name='genre_id_7').apply(''.join)

    for i in genre_id_7[genre_id_7 == ''].index:

        genre_id_7.iloc[i] = 'none'

    

    for i in genre_id.astype(str):

        num_genre_types.append(len(i.split(',')))

    num_genre_types = pd.Series(num_genre_types, name='num_genre_types').astype(int)

    df = pd.concat([df, genre_id_1, genre_id_2, genre_id_3, genre_id_4, genre_id_5, 

                    genre_id_6, genre_id_7, num_genre_types], axis=1)

    

    # Create column for sequels

    is_sequel = []

    for i in df['Keywords']:

        if 'sequel' in str(i):

            is_sequel.append(1)

        else:

            is_sequel.append(0)

    is_sequel = pd.Series(is_sequel, name='is_sequel')

    df = pd.concat([df, is_sequel], axis=1)

    

    keyword_words = []

    for i in df['Keywords']:

        i = re.findall('[a-zA-Z \t]+', str(i))

        stopwords = ['id', 'name', ' ']

        i = [word for word in i if word not in stopwords]

        keyword_words.append(i)

    keyword_words = pd.Series(keyword_words, name='keyword_words')

    df = pd.concat([df, keyword_words], axis=1)



    # This will count the number of Keywords listed for each film

    num_keywords = []

    for i in keyword_words:

        num_keywords.append(len(str(i).split(',')))

    num_keywords = pd.Series(num_keywords, name='num_keywords').astype(int)

    df = pd.concat([df, num_keywords], axis=1)

    

    # Create column for Keyword Id numbers

    keyword_ids = []

    for i in df['Keywords']:

        i = re.findall('[0-9]+', str(i))

        keyword_ids.append(i)

    keyword_ids = pd.Series(keyword_ids, name='keyword_ids')

    #df = pd.concat([keyword_ids, df], axis=1)

    

    # Extract number from belongs to collection

    collection_id = []

    for i in df['belongs_to_collection']:

        i = re.findall('[0-9]+', str(i))

        collection_id.append(i[:1])

    collection_id= pd.Series(collection_id, name='collection_id').apply(lambda x: ''.join([str(i) for i in x]))



    # Fill in blank values with 'No Collection'

    for i in collection_id[collection_id == ''].index:

        collection_id.loc[i] = 'no collection'

    collection_id = collection_id

    df = pd.concat([df, collection_id], axis=1)

    

    # Add column with 1 for movies in a collection and 0 if not

    is_in_collection = []

    for i in df['collection_id']:

        if i != 'no collection':

            is_in_collection.append(1)

        else:

            is_in_collection.append(0)

    is_in_collection = pd.Series(is_in_collection, name='is_in_collection').astype(int)

    df = pd.concat([is_in_collection, df], axis=1)

    

    # Create a column for production country (1 for US, 0 for rest of world)

    # It would be helpful if countries had different codes, but they all appear to be the same so it's difficult to work with

    US_prod_country = []

    for i in df['production_countries']:

        if 'US' in str(i):

            US_prod_country.append(1)

        else:

            US_prod_country.append(0)

    US_prod_country = pd.Series(US_prod_country, name='US_prod_country')

    df = pd.concat([df, US_prod_country], axis=1)

    

    # Create column for number of production countries

    num_prod_countries = []

    for i in df['production_countries']:

        i = re.findall('[a-zA-Z \t]+', str(i))

        num_prod_countries.append(str(i).count('name'))

    num_prod_countries = pd.Series(num_prod_countries, name='num_production_countries')

    df = pd.concat([df, num_prod_countries], axis=1)

    

    # Create a column for each production company name and a column for the number of companies

    production_company_names = []

    num_production_companies = []

    for i in df['production_companies']:

        i = re.findall('[a-zA-Z \t]+', str(i))

        stopwords = ['id', 'name', ' ']

        production_company_names.append([word for word in i if word not in stopwords])

        num_production_companies.append(str(i).count('name'))



    production_company_1 = []

    production_company_2 = []

    production_company_3 = []

    production_company_4 = []

    production_company_5 = []

    production_company_6 = []

    production_company_7 = []

    production_company_8 = []



    for i in production_company_names:

        try:

            production_company_1.append(i[:][0:1])

            production_company_2.append(i[:][1:2])

            production_company_3.append(i[:][2:3])

            production_company_4.append(i[:][3:4])

            production_company_5.append(i[:][4:5])

            production_company_6.append(i[:][5:6])

            production_company_7.append(i[:][6:7])

            production_company_8.append(i[:][7:8])

        except:

            production_company_1.append('none')

            production_company_2.append('none')

            production_company_3.append('none')

            production_company_4.append('none')

            production_company_5.append('none')

            production_company_6.append('none')

            production_company_7.append('none')

            production_company_8.append('none')



    num_production_companies = pd.Series(num_production_companies, name='num_production_companies')

    production_company_1 = pd.Series(production_company_1, name='production_company_1').apply(''.join)

    for i in production_company_1[production_company_1 == ''].index:

        production_company_1.iloc[i] = 'none'

    production_company_2 = pd.Series(production_company_2, name='production_company_2').apply(''.join)

    for i in production_company_2[production_company_2 == ''].index:

        production_company_2.iloc[i] = 'none'

    production_company_3 = pd.Series(production_company_3, name='production_company_3').apply(''.join)

    for i in production_company_3[production_company_3 == ''].index:

        production_company_3.iloc[i] = 'none'

    production_company_4 = pd.Series(production_company_4, name='production_company_4').apply(''.join)

    for i in production_company_4[production_company_4 == ''].index:

        production_company_4.iloc[i] = 'none'

    production_company_5 = pd.Series(production_company_5, name='production_company_5').apply(''.join)

    for i in production_company_5[production_company_5 == ''].index:

        production_company_5.iloc[i] = 'none'

    production_company_6 = pd.Series(production_company_6, name='production_company_6').apply(''.join)

    for i in production_company_6[production_company_6 == ''].index:

        production_company_6.iloc[i] = 'none'

    production_company_7 = pd.Series(production_company_7, name='production_company_7').apply(''.join)

    for i in production_company_7[production_company_7 == ''].index:

        production_company_7.iloc[i] = 'none'

    production_company_8 = pd.Series(production_company_8, name='production_company_8').apply(''.join)

    for i in production_company_8[production_company_8 == ''].index:

        production_company_8.iloc[i] = 'none'

    df = pd.concat([df, num_production_companies, production_company_1, production_company_2,

              production_company_3, production_company_4, production_company_5, production_company_6,

              production_company_7, production_company_8], axis=1)

    

    # Create a column for number of spoken languages

    num_spoken_languages=[]

    for i in df['spoken_languages']:

        a = str(i).split()

        num_spoken_languages.append(a.count("'name':"))

    num_spoken_languages = pd.Series(num_spoken_languages, name = 'num_spoken_languages')

    df = pd.concat([df, num_spoken_languages], axis=1)

        

    # Create column for release status

    status_is_released = []

    for i in df['status']:

        if i == 'Released':

            status_is_released.append(1)

        else:

            status_is_released.append(0)

    status_is_released = pd.Series(status_is_released, name = 'status_is_released')

    df = pd.concat([df, status_is_released], axis=1)

    

    # Drop columns that have been engineered

    df = df.drop(['belongs_to_collection', 'genres', 'Keywords', 'belongs_to_collection', 'homepage', 'imdb_id', 

                 'original_title', 'overview', 'poster_path', 'production_companies', 'production_countries',

                 'release_date', 'spoken_languages', 'status', 'tagline', 'title', 'cast', 'crew'], axis=1)

    # Drop 'keyword_words' column for now.  Can work with it later.

    df = df.drop(['keyword_words'], axis=1)

    return reduce_mem_usage(df)
# Reload the data fresh and apply the processing function

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train = data_processing(train)

test = data_processing(test)
def train_target_encoded_year(df, cols):

    """Function will take a dataframe and replace any passed categorical columns with the average revenue for each unique value from a given year."""

    for i in cols:

        d = df.groupby(['release_year', i]).agg({'revenue':'mean'})

        df = df.set_index(['release_year', i], drop=False)

        df[i] = d.revenue

        df = df.reset_index(drop=True)

    return df
def test_target_encoded_year(df_train, df_test, cols):

    """Function will take a dataframe and replace any passed categorical columns with the unique average revenue per year generated from the training dataframe."""

    for i in cols:

        d = df_train.groupby(['release_year', i]).agg({'revenue':'mean'})

        df_test = df_test.set_index(['release_year', i], drop=False)

        df_test[i] = d.revenue

        df_test = df_test.reset_index(drop=True)

    return df_test
def target_encode(df, target_feature, m = 300): 

    d = defaultdict()

    target_mean = df[target_feature].mean()

    

    # Map values and create dictionary   

    for cat_feature in df.select_dtypes(include='category'):

        group_target_mean = df.groupby([cat_feature])[target_feature].agg('mean')

        group_target_count = df.groupby([cat_feature])[target_feature].agg('count')

        smooth = (group_target_count * group_target_mean + m * target_mean) / (group_target_count + m)

        k = pd.Series(df[cat_feature])

        v = df[cat_feature].map(smooth)

        d[cat_feature] = dict(zip(k, v))

        df[cat_feature] = df[cat_feature].map(smooth)

        

    return df, d

def test_target_encoded_year(df_train, df_test):

    """Function will take a dataframe and replace any passed categorical columns with the unique average revenue per year generated from the training dataframe."""

    cols = df_test.select_dtypes(include='object').columns

    for col in cols:

        d = df_train.groupby(['release_year', col]).agg({'revenue':'mean'})

        df_test = df_test.set_index(['release_year', col], drop=False)

        df_test[col] = d.revenue

        df_test = df_test.reset_index(drop=True)

        

    # There are a numerous missing values in the test set after processing so I'll fill them with the yearly avg.

    for col in cols:

        #d = defaultdict()

        X = df_test[df_test[col].notnull()]

        year_mean = pd.Series(X.groupby(['release_year'])[col].agg('mean'))

        d = dict(year_mean)

    

        for i in df_test[df_test['budget'].isnull()]:

            df_test[col] = df_test.loc[:, 'release_year'].map(d)

    

    return reduce_mem_usage(df_test)
# The numeric columns look okay, but budget may need normalization as the st. dev is quite large

train.describe()
from category_encoders import *

from sklearn.preprocessing import LabelEncoder
# Make complete list of genre ids

genre_ids = train['genre_id_1']

for i in train.loc[:, 'genre_id_2': 'genre_id_7'].columns:

    genre_ids = pd.concat([genre_ids, train[i]], axis=0)



le = LabelEncoder()

lab_enc = le.fit_transform(genre_ids)

genre_ids_dict = dict(zip(genre_ids, lab_enc))



# Map genre_ids_dict to genre_id columns

for i in train.loc[:, 'genre_id_1': 'genre_id_7'].columns:

    train[i] = train[i].map(genre_ids_dict)
train.loc[:, 'production_company_2': 'production_company_8'].head()
# Make complete list of production companies

prod_companies = train['production_company_1']

for i in train.loc[:, 'production_company_2': 'production_company_8'].columns:

    prod_companies = pd.concat([prod_companies, train[i]], axis=0)



le = LabelEncoder()

lab_enc = le.fit_transform(prod_companies)

prod_companies_dict = dict(zip(prod_companies, lab_enc))



# Map genre_ids_dict to genre_id columns

for i in train.loc[:, 'production_company_1': 'production_company_8'].columns:

    train[i] = train[i].map(prod_companies_dict)
le = LabelEncoder()

train['collection_id'] = le.fit_transform(train['collection_id'])

train['original_language'] = le.fit_transform(train['original_language'])
from sklearn.preprocessing import LabelEncoder

def cat_encode(df):

    le = LabelEncoder()

    

    # Make complete list of genre ids

    genre_ids = df['genre_id_1']

    for i in df.loc[:, 'genre_id_2': 'genre_id_7'].columns:

        genre_ids = pd.concat([genre_ids, df[i]], axis=0)



    lab_enc_genres = le.fit_transform(genre_ids)

    genre_ids_dict = dict(zip(genre_ids, lab_enc_genres))



    # Map genre_ids_dict to genre_id columns

    for i in df.loc[:, 'genre_id_1': 'genre_id_7'].columns:

        df[i] = df[i].map(genre_ids_dict)



    # Make complete list of production companies

    prod_companies = df['production_company_1']

    for i in df.loc[:, 'production_company_2': 'production_company_8'].columns:

        prod_companies = pd.concat([prod_companies, df[i]], axis=0)



    lab_enc_comp = le.fit_transform(prod_companies)

    prod_companies_dict = dict(zip(prod_companies, lab_enc_comp))



    # Map genre_ids_dict to genre_id columns

    for i in df.loc[:, 'production_company_1': 'production_company_8'].columns:

        df[i] = df[i].map(prod_companies_dict)

        

    df['collection_id'] = le.fit_transform(df['collection_id'])

    df['original_language'] = le.fit_transform(df['original_language'])

    

    return reduce_mem_usage(df)
train = cat_encode(train)

test = cat_encode(test)
train.shape
# Get an idea of what correlates most strongly with revenue

for i in train.columns:

    print(i, stats.pearsonr(train[i], train['revenue']))
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
X = train.drop(['id', 'revenue'], axis=1)

y = train['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)
sns.distplot((y_test-pred),bins=50)

plt.show()
def rmsle(y_true, y_pred):

    return 'rmsle', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

print('RMSLE:', rmsle(y_test, pred))
from lightgbm import LGBMRegressor
lr = LGBMRegressor(boosting_type='dart',num_leaves=20,max_depth=-1,min_data_in_leaf=20, learning_rate=0.2,n_estimators=500,subsample_for_bin=200000,

                   class_weight=None,min_split_gain=0.0,min_child_weight=0.001,subsample=0.1,subsample_freq=0,colsample_bytree=0.75,reg_alpha=0.0,reg_lambda=0.0,

                   random_state=101,n_jobs=-1)
lr.fit(X_train, y_train,eval_set=[(X_test, y_test)],eval_metric=rmsle)

pred = lr.predict(X_test, num_iteration=lr.best_iteration_)
print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

print('RMSLE:', rmsle(y_test, pred))
submission = pd.DataFrame()

submission['id'] = test['id']

submission['revenue'] = lr.predict(test.drop('id', axis=1), num_iteration=lr.best_iteration_)
submission.to_csv('TMDB_test_predictions.csv', index=False)