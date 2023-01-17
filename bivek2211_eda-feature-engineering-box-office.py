# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load packages



import sys              # access to system parameters https://docs.python.org/3/library/sys.html

import pandas as pd        # collection of functions for data processing and analysis modeled after R dataframes with SQL like features

import matplotlib as mpl     # collection of fun for scientific and publication-ready visualization

import numpy as np                 # foundational package for scientific computing

import scipy as sp                 # collection of functions for scientific computing and advance mathematics

import IPython

from IPython import display        # pretty printing of dataframes in Jupyter notebook

import sklearn                     # collection of machine learning algorithms)

import plotly.graph_objs as go

from scipy.stats import norm,skew  # for some statistics

from wordcloud import WordCloud

from collections import Counter

from scipy import stats

import ast

import plotly.offline as py

py.init_notebook_mode(connected=True) # for make plot as notebook editable





#misc libraries

import random

import time



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('always')



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print(train_df.shape)

print(test_df.shape)
## memory reducer
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df



train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)
train_df.isna().sum()
# transforming dictionary columns to proper format( Nan to {})

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train_df = text_to_dict(train_df)

test_df = text_to_dict(test_df)
train_df.info()
# removing outliers 



# movies with budget and runtime 0

train_df.drop(train_df[train_df['budget'] == 0].index, inplace=True)

train_df.drop(train_df[train_df['runtime'] == 0].index, inplace=True)
## looking all movies with runlength > 3.5hrs

train_df.loc[train_df['runtime'].fillna(0) / 60 > 3.5 ]
## belongs_to_collection
for i,j in enumerate(train_df.belongs_to_collection[:6]):

    print(i,j) 
a=b=0

for j in train_df.belongs_to_collection:

    if j != {}: 

        a=a+1

    else:

        b=b+1

print(f"len not 0 :{a}    len = 0 :{b} ")
train_df['collection_name'] = train_df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x!= {} else 0)

train_df['has_collection'] = train_df['belongs_to_collection'].apply(lambda x: len(x) if x!={} else 0)



test_df['collection_name'] = test_df['belongs_to_collection'].apply(lambda x: x[0]['name'] if x!= {} else 0)

test_df['has_collection'] = test_df['belongs_to_collection'].apply(lambda x: len(x) if x!={} else 0)



train_df.drop('belongs_to_collection', axis =1, inplace= True)

test_df.drop('belongs_to_collection', axis =1, inplace= True)
sns.swarmplot(x='has_collection', y='revenue', data=train_df);

plt.title('Revenue for film with and without collection');
# there are lot's of movies with doesn't have collection but earn large revenues
## genres
for i,j in enumerate(train_df.genres[:10]):

    print(f"{i} {j}")
## finding list of genres

print('Number of genres in films')

train_df['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_genres = list(train_df['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
## finding top genres

plt.figure(figsize = (12, 8))

text = ' '.join([i for j in list_of_genres for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(text)



plt.imshow(wordcloud)

plt.title('Top genres')

plt.axis("off")

plt.show()
Counter([i for j in list_of_genres for i in j]).most_common()[:4]
## creating seperate column for top 15 genre



train_df['num_genres'] = train_df['genres'].apply(lambda x: len(x) if x != {} else 0)

train_df['all_genres'] = train_df['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]

for g in top_genres:

    train_df['genre_' + g] = train_df['all_genres'].apply(lambda x: 1 if g in x else 0)

    

test_df['num_genres'] = test_df['genres'].apply(lambda x: len(x) if x != {} else 0)

test_df['all_genres'] = test_df['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_genres:

    test_df['genre_' + g] = test_df['all_genres'].apply(lambda x: 1 if g in x else 0)

    

train_df = train_df.drop(['genres'], axis=1)

test_df = test_df.drop(['genres'], axis=1)
## production_companies
for i,j in enumerate(train_df.production_companies[:5]):

    print(i,j)
print('Number of production companies in films')

train_df['production_companies'].apply(lambda x: len(x) if x != {} else 0).value_counts()[:5]
# lets look movies with more than 10 production companies

train_df[train_df['production_companies'].apply(lambda x: len(x) if x != {} else 0) > 10].head()
## finding list of companies



list_of_companies = list(train_df['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

list_of_companies[:5]
## taking top 30 diffrent companies

Counter([i for j in list_of_companies for i in j]).most_common(30)[:5]
## binary col for top 20 production companies



train_df['num_companies'] = train_df['production_companies'].apply(lambda x: len(x) if x != {} else 0)

train_df['all_production_companies'] = train_df['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]

for g in top_companies:

    train_df['production_company_' + g] = train_df['all_production_companies'].apply(lambda x: 1 if g in x else 0)

    

test_df['num_companies'] = test_df['production_companies'].apply(lambda x: len(x) if x != {} else 0)

test_df['all_production_companies'] = test_df['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_companies:

    test_df['production_company_' + g] = test_df['all_production_companies'].apply(lambda x: 1 if g in x else 0)



    

train_df = train_df.drop(['production_companies', 'all_production_companies'], axis=1)

test_df = test_df.drop(['production_companies', 'all_production_companies'], axis=1)
## production_countries
for i, e in enumerate(train_df['production_countries'][:5]):

    print(i, e)
print('Number of production countries in films')

train_df['production_countries'].apply(lambda x: len(x) if x != {} else 0).value_counts().head()
list_of_countries = list(train_df['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



Counter([i for j in list_of_countries for i in j]).most_common(25)[:5]
train_df['num_countries'] = train_df['production_countries'].apply(lambda x: len(x) if x != {} else 0)

train_df['all_countries'] = train_df['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]

for g in top_countries:

    train_df['production_country_' + g] = train_df['all_countries'].apply(lambda x: 1 if g in x else 0)

    

test_df['num_countries'] = test_df['production_countries'].apply(lambda x: len(x) if x != {} else 0)

test_df['all_countries'] = test_df['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_countries:

    test_df['production_country_' + g] = test_df['all_countries'].apply(lambda x: 1 if g in x else 0)



    

train_df = train_df.drop(['production_countries', 'all_countries'], axis=1)

test_df = test_df.drop(['production_countries', 'all_countries'], axis=1)
## spoken_languages



for i, e in enumerate(train_df['spoken_languages'][:5]):

    print(i, e)
print('Number of spoken languages in films')

train_df['spoken_languages'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_languages = list(train_df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_languages for i in j]).most_common(30)[:5]
train_df['num_languages'] = train_df['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

train_df['all_languages'] = train_df['spoken_languages'].apply(lambda x: ' '.join(sorted([i['iso_639_1'] for i in x])) if x != {} else '')

top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]

for g in top_languages:

    train_df['language_' + g] = train_df['all_languages'].apply(lambda x: 1 if g in x else 0)

    

test_df['num_languages'] = test_df['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

test_df['all_languages'] = test_df['spoken_languages'].apply(lambda x: ' '.join(sorted([i['iso_639_1'] for i in x])) if x != {} else '')

for g in top_languages:

    test_df['language_' + g] = test_df['all_languages'].apply(lambda x: 1 if g in x else 0)



    

train_df = train_df.drop(['spoken_languages', 'all_languages'], axis=1)

test_df = test_df.drop(['spoken_languages', 'all_languages'], axis=1)
## keywords
for i, e in enumerate(train_df['Keywords'][:5]):

    print(i, e)
print('Number of Keywords in films')

train_df['Keywords'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
list_of_keywords = list(train_df['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

plt.figure(figsize = (16, 12))

text = ' '.join(['_'.join(i.split(' ')) for j in list_of_keywords for i in j])

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(text)



plt.imshow(wordcloud)

plt.title('Top keywords')

plt.axis("off")

plt.show()
train_df['num_Keywords'] = train_df['Keywords'].apply(lambda x: len(x) if x != {} else 0)

train_df['all_Keywords'] = train_df['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]

for g in top_keywords:

    train_df['keyword_' + g] = train_df['all_Keywords'].apply(lambda x: 1 if g in x else 0)

    

test_df['num_Keywords'] = test_df['Keywords'].apply(lambda x: len(x) if x != {} else 0)

test_df['all_Keywords'] = test_df['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_keywords:

    test_df['keyword_' + g] = test_df['all_Keywords'].apply(lambda x: 1 if g in x else 0)



    

train_df = train_df.drop(['Keywords', 'all_Keywords'], axis=1)

test_df = test_df.drop(['Keywords', 'all_Keywords'], axis=1)
## cast
for i, e in enumerate(train_df['cast'][:1]):

    print(i, e)
print('Number of casted persons in films')

train_df['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
## taking most common names...top 15

list_of_cast_names = list(train_df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_names for i in j]).most_common(15)[:5]
train_df['num_cast'] = train_df['cast'].apply(lambda x: len(x) if x != {} else 0)

top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]

for g in top_cast_names:

    train_df['cast_name_' + g] = train_df['cast'].apply(lambda x: 1 if g in x else 0)

    

test_df['num_cast'] = test_df['cast'].apply(lambda x: len(x) if x != {} else 0)

for g in top_cast_names:

    test_df['cast_name_' + g] = test_df['cast'].apply(lambda x: 1 if g in x else 0)

    

train_df = train_df.drop(['cast'], axis=1)

test_df = test_df.drop(['cast'], axis=1)
## release date
test_df.loc[test_df['release_date'].isnull() == True, 'release_date'] = '01/01/98'
def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year
train_df['release_date'] = train_df['release_date'].apply(lambda x: fix_date(x))

test_df['release_date'] = test_df['release_date'].apply(lambda x: fix_date(x))



train_df['release_date'] = pd.to_datetime(train_df['release_date'])

test_df['release_date'] = pd.to_datetime(test_df['release_date'])
# creating features based on dates

def process_date(df):

    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        part_col = 'release_date' + "_" + part

        df[part_col] = getattr(df['release_date'].dt, part).astype(int)

    

    return df



train_df = process_date(train_df)

test_df = process_date(test_df)



train_df.drop('release_date',axis=1, inplace=  True)

test_df.drop('release_date',axis=1, inplace=  True)
d1 = train_df['release_date_year'].value_counts().sort_index()

d2 = test_df['release_date_year'].value_counts().sort_index()

data = [go.Scatter(x=d1.index, y=d1.values, name='train'), go.Scatter(x=d2.index, y=d2.values, name='test')]

layout = go.Layout(dict(title = "Number of films per year",

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Count'),

                  ),legend=dict(

                orientation="v"))

py.iplot(dict(data=data, layout=layout))
d1 = train_df['release_date_year'].value_counts().sort_index()

d2 = train_df.groupby(['release_date_year'])['revenue'].mean()

data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), go.Scatter(x=d2.index, y=d2.values, name='mean revenue', yaxis='y2')]

layout = go.Layout(dict(title = "Number of films and average revenue per year",

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Count'),

                  yaxis2=dict(title='Average revenue', overlaying='y', side='right')

                  ),legend=dict(

                orientation="v"))

py.iplot(dict(data=data, layout=layout))

# The number of films and total revenue are growing, which is to be expected. But there were some years in the past with a high number of successful films, which brought high revenue.
sns.swarmplot(x='release_date_weekday', y='revenue', data=train_df);

plt.title('Revenue on different days of week of release');
## surprisingly films releases on Wednesdays and on Thursdays tend to have a higher revenue

## also, there is large %age of movies released on Friday
sns.swarmplot(x='release_date_month', y='revenue', data=train_df);

plt.title('Revenue on different days of week of release');
## jan and aug are not that great for movies as compared to other months
## crew



train_df.drop('crew', axis = 1, inplace= True)

test_df.drop('crew', axis =1, inplace = True)
## released
train_df.status.value_counts()
## let's check those rumored movies



train_df.loc[train_df.status == "Rumored"]
train_df.drop(['imdb_id','status'], axis = 1, inplace=True)

test_df.drop(['imdb_id','status'], axis = 1, inplace=True)
## revenue
fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.hist(train_df['revenue']);

plt.title('Distribution of revenue');



plt.subplot(1, 2, 2)

plt.hist(np.log1p(train_df['revenue']));

plt.title('Distribution of log of revenue');
train_df['revenue'] = np.log1p(train_df['revenue'])
## homepage 
train_df['has_homepage'] = 0

train_df.loc[train_df['homepage'].isnull() == False, 'has_homepage'] = 1

test_df['has_homepage'] = 0

test_df.loc[test_df['homepage'].isnull() == False, 'has_homepage'] = 1
sns.catplot(x='has_homepage', y='revenue', data=train_df);

plt.title('Revenue for film with and without homepage');
# films with homepage likely to generate more revenue

train_df.has_homepage.value_counts()
train_df.drop('homepage', axis=1, inplace=True)

test_df.drop('homepage', axis=1, inplace=True)
## poster_path
train_df['has_posterpath'] = 0

train_df.loc[train_df['poster_path'].isnull() == False, 'has_posterpath'] = 1

test_df['has_posterpath'] = 0

test_df.loc[test_df['poster_path'].isnull() == False, 'has_posterpath'] = 1
sns.catplot(x='has_posterpath', y='revenue', data=train_df);

plt.title('Revenue for film with and without poster');
train_df.has_posterpath.value_counts()

## every one has poster path so we can drop col
train_df.drop(['poster_path','has_posterpath'], axis=1, inplace= True)

test_df.drop(['poster_path','has_posterpath'], axis=1, inplace= True)
## original_language
# analysing top 10 languages

fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

sns.boxplot(x='original_language', y='revenue',data=train_df.loc[train_df['original_language'].isin(train_df['original_language'].value_counts().head(10).index)])

plt.title('Distribution of language');



plt.subplot(1, 2, 2)

sns.countplot(data=train_df.loc[train_df['original_language'].isin(train_df['original_language'].value_counts().head(10).index)], x='original_language')

plt.title('Count of Language')
## majority of the movie are in english with higher revenues but there are other languages too with higher revenues 
train_df.original_language.value_counts()
## original_title
plt.figure(figsize = (12, 12))

text = ' '.join(train_df['original_title'].values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top words in titles')

plt.axis("off")

plt.show()
# counting the title of movie having word "Man"

train_df['original_title'].apply(lambda x: 1 if "Man" in x else 0).value_counts()

# checking th title of movie having word "Man"

# train_df['original_title'].apply(lambda x: print(x) if "Man" in x else 0)
## overview
plt.figure(figsize = (12, 12))

text = ' '.join(train_df['overview'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(wordcloud)

plt.title('Top words in overview')

plt.axis("off")

plt.show()
## popularity
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(x= train_df['popularity'],y= train_df['revenue'])

plt.title('Revenue vs popularity');
## runtime



plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)

plt.hist(train_df['runtime'].fillna(0) / 60, bins=40);

plt.title('Distribution of length of film in hours');



plt.subplot(1, 3, 2)

plt.scatter(train_df['runtime'].fillna(0), train_df['revenue'])

plt.title('runtime vs revenue');



plt.subplot(1, 3, 3)

plt.scatter(train_df['runtime'].fillna(0), train_df['popularity'])

plt.title('runtime vs popularity');
## genres

f, axes = plt.subplots(3, 5, figsize=(30, 15))

plt.suptitle('Violinplot of revenue vs genres')

for i, e in enumerate([col for col in train_df.columns if 'genre_' in col]):

    sns.violinplot(x=e, y='revenue', data=train_df, ax=axes[i // 5][i % 5]);
## basic modelling
train_df.drop(['collection_name','all_genres'], axis=1 ,inplace= True)

test_df.drop(['collection_name','all_genres'], axis=1 ,inplace= True)
for col in train_df.columns:

    if train_df[col].nunique() == 1:

        print(col)

        train_df = train_df.drop([col], axis=1)

        test_df = test_df.drop([col], axis=1)
for col in ['original_language']:

    le = LabelEncoder()

    le.fit(list(train_df[col].fillna('')) + list(test_df[col].fillna('')))

    train_df[col] = le.transform(train_df[col].fillna('').astype(str))

    test_df[col] = le.transform(test_df[col].fillna('').astype(str))
train_df.sample(3)
train_df.to_csv('train_cleaned.csv',index=False)

test_df.to_csv('test_cleaned.csv',index=False)