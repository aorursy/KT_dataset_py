import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.



!pip install fastai==0.7.0 --quiet



import datetime as dt

from fastai.structured import add_datepart
from scipy import stats
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.shape, test.shape
train.describe()
train.isna().sum()
train.columns
## Converting columns into usable form



import ast



for column in ['belongs_to_collection','genres', 'production_companies','production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']:

    train[column] = train[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x!={} else 0)

train['has_collection'] = train['belongs_to_collection'].apply(lambda x: 1 if x!={} else 0)



train = train.drop(['belongs_to_collection'], axis=1)
train
train['has_homepage'] = train['homepage'].apply(lambda x: 0 if pd.isna(x) else 1)



train = train.drop(['homepage'], axis=1)
train['num_genre'] = train['genres'].apply(lambda x: len(x) if x!= {} else 0)
train['all_genre'] = train['genres'].apply(lambda x: ' '.join([e['name'] for e in x]) if x!= {} else '')



train['all_genre'] = list(train['all_genre'].apply(lambda x: x.split(" ") if x!='' else [] ))



list_all_genre = list(train['all_genre'])



list_of_genres = list(set([genre for list_gen in list_all_genre for genre in list_gen]))



list_of_genres = ['gen_'+ s for s in list_of_genres]



train = train.reindex( columns = train.columns.tolist() + list_of_genres)



for i, e in train.iterrows():

    for genre in (e['all_genre']):

        #print(genre)

        #print(train.loc[i,('gen_'+genre)])

        train.loc[i,('gen_'+genre)] =1



train[list_of_genres] = train[list_of_genres].fillna(0)

train = train.drop(['all_genre','genres'], axis=1)
train
train['production_companies_name'] = train['production_companies'].apply(lambda x: ','.join([e['name'] for e in x]) if x!={} else '')



train['production_companies_name'] = train['production_companies_name'].apply(lambda x: x.split(",") if x!='' else [] )



#train['production_companies_name']



companies_count = train['production_companies_name'].apply(pd.Series).stack().value_counts()

common_prod_companies = companies_count[companies_count> 30].keys()



common_prod_companies = ['comp_'+ s for s in list(common_prod_companies)]

train = train.reindex( columns = train.columns.tolist() + common_prod_companies)



train.head()
for i, e in train.iterrows():

    for comp in (e['production_companies_name']):

        if 'comp_'+ comp in list(common_prod_companies):

            train.loc[i,('comp_'+comp)] =1



train[list(common_prod_companies)] = train[list(common_prod_companies)].fillna(0)
train = train.drop(['production_companies_name','production_companies'], axis=1)
train['production_countries_name'] = train['production_countries'].apply(lambda x: ','.join([e['name'] for e in x]) if x!={} else '')



train['production_countries_name'] = train['production_countries_name'].apply(lambda x: x.split(",") if x!='' else [] )



#train['production_countries_name']



countries_count = train['production_countries_name'].apply(pd.Series).stack().value_counts()

common_countries = countries_count[countries_count> 30].keys()



common_countries = ['country_'+ s for s in list(common_countries)]



train = train.reindex( columns = train.columns.tolist() + list(common_countries))



for i, e in train.iterrows():

    for count in (e['production_countries_name']):

        if 'country_'+count in list(common_countries):

            train.loc[i,'country_'+count] =1





train[list(common_countries)] = train[list(common_countries)].fillna(0)
train = train.drop(['production_countries', 'production_countries_name'], axis=1)
train.head()
train['all_spoken_languages'] = train['spoken_languages'].apply(lambda x: ','.join([e['name'] for e in x]) if x!={} else '')



train['no_of_languages_spoken'] = train['all_spoken_languages'].apply(lambda x: len(x.split(",")))
train = train.drop(['all_spoken_languages'], axis=1)
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



Counter([i for j in list_of_cast_names for i in j]).most_common(15)



top_cast_names = [e[0] for e in (Counter([i for j in list_of_cast_names for i in j]).most_common(15))]

 

top_cast_names = ['cast_name_' + s for s in top_cast_names]



train = train.reindex( columns = train.columns.tolist() + top_cast_names)



for i, e in train.iterrows():

    for cast in (e['cast']):

        if 'cast_name_'+cast['name'] in top_cast_names:

            train.loc[i,('cast_name_'+ cast['name'])] = 1





train[top_cast_names] = train[top_cast_names].fillna(0)

list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



Counter([i for j in list_of_crew_names for i in j]).most_common(15)



top_crew_names = [e[0] for e in (Counter([i for j in list_of_crew_names for i in j]).most_common(15))]



top_crew_names = ['crew_name_'+s for s in top_crew_names]



train = train.reindex( columns = train.columns.tolist() + top_crew_names)



for i, e in train.iterrows():

    for crew in (e['crew']):

        if ('crew_name_'+crew['name']) in top_crew_names:

            train.loc[i, ('crew_name_'+crew['name'])] = 1





train[top_crew_names] = train[top_crew_names].fillna(0)
list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)



Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)



top_crew_jobs = [e[0] for e in (Counter([i for j in list_of_crew_jobs for i in j]).most_common(15))]



top_crew_jobs = ['crew_jobs_'+ s for s in top_crew_jobs]



train = train.reindex( columns = train.columns.tolist() + top_crew_jobs)



for i, e in train.iterrows():

    for crew in (e['crew']):

        if ('crew_jobs_'+crew['job']) in top_crew_jobs:

            train.loc[i,('crew_jobs_'+crew['job'])] = 1





train[top_crew_jobs] = train[top_crew_jobs].fillna(0)
list_of_crew_depts = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)



Counter([i for j in list_of_crew_depts for i in j]).most_common(15)



top_crew_depts = [e[0] for e in (Counter([i for j in list_of_crew_depts for i in j]).most_common(15))]



top_crew_depts = ['crew_dept_'+s for s in top_crew_depts]



train = train.reindex( columns = train.columns.tolist() + top_crew_depts)



for i, e in train.iterrows():

    for crew in (e['crew']):

        if ('crew_dept_'+crew['department']) in top_crew_depts:

            train.loc[i,('crew_dept_'+crew['department'])] = 1



train[top_crew_depts] = train[top_crew_depts].fillna(0)
train['release_date'] = pd.to_datetime(train['release_date'])



train['release_date']



add_datepart(train, 'release_date')

train.dtypes
train
train['Keywords'][0]
train['all_keywords'] = train['Keywords'].apply(lambda x: ','.join([e['name'] for e in x]) if x!={} else '')
train
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)





Counter([i for j in list_of_keywords for i in j]).most_common(15)



top_keywords = [e[0] for e in (Counter([i for j in list_of_keywords for i in j]).most_common(15))]



top_keywords = ['keywords_'+s for s in top_keywords]



train = train.reindex( columns = train.columns.tolist() + top_keywords)



for i, e in train.iterrows():

    for key in (e['Keywords']):

        if ('keywords_'+key['name']) in top_keywords:

            train.loc[i,('keywords_'+key['name'])] = 1





train[top_keywords] = train[top_keywords].fillna(0)
train['collection_name'].value_counts()
plt.figure(figsize=(100,10))

train['collection_name'].value_counts().plot(kind='bar')
train['has_collection'].value_counts()
train['has_collection'].value_counts().plot(kind='bar')
sns.regplot(x=train['has_collection'], y=train['revenue'], fit_reg=False)
sns.distplot(train['revenue'])
train['box_cox_revenue'],fitted_lambda = stats.boxcox(train['revenue'])
sns.distplot(train['box_cox_revenue'])
from numpy import log

train['log_revenue'] = log(train['revenue'])
sns.distplot(train['log_revenue'])
sns.distplot(train['budget'])
train['log_budget'] = np.log1p(train['budget'])
sns.distplot(train['log_budget'])
sns.regplot(x= train['budget'], y=train['revenue'], fit_reg = False)
sns.regplot(x= train['log_budget'], y=train['log_revenue'], fit_reg = False)
list_all_genre
all_genres = Counter([genre for list_gen in list_all_genre for genre in list_gen])
keys  = Counter([genre for list_gen in list_all_genre for genre in list_gen]).keys()
count = [all_genres[k] for k in keys] 
plt.figure(figsize=(10,10))

plt.xlabel("Count of each genre")

plt.ylabel("Genre type")

plt.barh(list(keys), list(count))
f, axes = plt.subplots(5, 5, figsize= (24,32))

for i, e in enumerate([col for col in train.columns if 'gen_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
from wordcloud import WordCloud, STOPWORDS
text_for_genre = ' '.join([genre for list_gen in list_all_genre for genre in list_gen])
genre_cloud = WordCloud(width= 1000, height=1000, background_color='white', min_font_size=10, collocations=False).generate(text_for_genre)
plt.figure(figsize=(8, 8))

plt.imshow(genre_cloud)

train['has_homepage'].value_counts()
train['has_homepage'].value_counts().plot(kind='bar')

plt.xlabel("Has Homepage or not")

plt.ylabel("Count for each category")

plt.show()
sns.regplot(x=train['has_homepage'], y=train['revenue'], fit_reg=False)
train['original_language'].value_counts()
train[['original_language','revenue','log_revenue']].groupby('original_language').agg(['min','max','mean'])
plt.figure(figsize=(16,8))

sns.boxplot(x='original_language', y='revenue', data=train);

plt.show()
plt.figure(figsize=(16,8))

sns.boxplot(x='original_language', y='log_revenue', data=train);

plt.show()
train['overview'].fillna('None', inplace=True)
overview_text  = ' '.join(e for e in train['overview'])
stopwords = set(STOPWORDS)
overview_cloud  = WordCloud(width= 1000, height = 1000, background_color ='white', max_font_size=None).generate(overview_text)
plt.figure(figsize=(12,12))

plt.imshow(overview_cloud)
print("Popularity min is {}, mean is {}, max is {}".format(train['popularity'].min(), train['popularity'].mean(),train['popularity'].max() ))
plt.figure(figsize=(12,12))

sns.boxplot(x='popularity', data = train, orient='v')
sns.regplot(x=train['popularity'], y=train['revenue'], fit_reg=False)
f, axes = plt.subplots(4, 5, figsize= (24,32))

for i, e in enumerate([col for col in train.columns if 'comp_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
f, axes = plt.subplots(3, 5, figsize= (24,32))

for i, e in enumerate([col for col in train.columns if 'country_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
f, axes = plt.subplots(3, 5, figsize= (24,32))

for i, e in enumerate([col for col in train.columns if 'keywords_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
f, axes = plt.subplots(3, 5, figsize= (24,32))

for i, e in enumerate([col for col in train.columns if 'cast_name_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
f, axes = plt.subplots(3, 5, figsize= (24,32))

for i, e in enumerate([col for col in train.columns if 'crew_name_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
f, axes = plt.subplots(3, 5, figsize= (24,32))

for i, e in enumerate([col for col in train.columns if 'crew_dept_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
f, axes = plt.subplots(3, 5, figsize= (24,32))

for i, e in enumerate([col for col in train.columns if 'crew_jobs_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
sns.regplot(x='runtime', y='revenue', data=train, fit_reg=False)
train['status']
train['status'] = train.status.astype("category").cat.codes
sns.regplot(x='status', y='revenue', data=train, fit_reg=False)
train.columns
mostly_release_year = train['release_Year'].value_counts()

mostly_release_year_movies = mostly_release_year[mostly_release_year>30].keys()
# plt.figure(figsize=(24,24))

# ax = sns.boxplot(x='release_Year', y='revenue', data=train[train['revenue'] in list(mostly_release_year_movies)])

# for label in ax.xaxis.get_ticklabels():

#     label.set_rotation(60)
train['release_Year'].unique()
print(train.columns)
for col in train.columns:

    print(col)
#print(train['release_Dayofweek'])

train[['revenue','release_Dayofweek']].groupby('release_Dayofweek').agg('mean')