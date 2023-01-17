import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split, KFold

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import StandardScaler

stop = set(stopwords.words('english'))

import os

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

import lightgbm as lgb

from sklearn import model_selection

from sklearn.metrics import accuracy_score

import json

import ast

import eli5

import shap

from catboost import CatBoostRegressor

from urllib.request import urlopen

from PIL import Image

from sklearn.preprocessing import LabelEncoder

import time

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn import linear_model

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')



train.info()
test = pd.read_csv('../input/test.csv')

test.info()
train.head()
test.head()
train.describe()
# load data
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df



train = text_to_dict(train)

print(train.head())

test = text_to_dict(test)

# print(test.head())
# Train数据中只有3000个样本！希望这足够训练模特。



# 我们可以看到一些列包含有字典的列表。有些列表包含一个字典，有些则包含几个然后我们从这些列中提取数据！
for i, e in enumerate(train['belongs_to_collection'][:5]):

    print(i, e)
train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0).value_counts()


train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



train = train.drop(['belongs_to_collection'], axis=1)

test = test.drop(['belongs_to_collection'], axis=1)
# print(train['collection_name'])

# print(train['has_collection'])
for i, e in enumerate(train['genres'][:5]):

    print(i, e)
print('Number of genres in films')

genres_count=train['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()

genres_count
# 类型栏包含电影所属类型的名称和ID。大多数电影有2-3个体裁，5-6个体裁是可能的。

# 我想0和7是异常值。让我们提取体裁！我将创建一个包含电影中所有类型的专栏，并为每个类型分开专栏。

#首先让我们来看看流派本身。
list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

list_of_genres
Counter([i for j in list_of_genres for i in j]).most_common()
train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)

train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')



top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]

for g in top_genres:

    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)

    

test['num_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)

test['all_genres'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_genres:

    test['genre_' + g] = test['all_genres'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['genres'], axis=1)

test = test.drop(['genres'], axis=1)
for i, e in enumerate(train['production_companies'][:5]):

    print(i, e)
print('Number of production companies in films')

train['production_companies'].apply(lambda x: len(x) if x != {} else 0).value_counts()
train[train['production_companies'].apply(lambda x: len(x) if x != {} else 0) > 11]
# 现在我不知道如何处理这些数据。我只需要为前30部电影创建二进制专栏
list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_companies for i in j]).most_common(30)
for i, e in enumerate(train['production_countries'][:5]):

    print(i, e)
print('Number of production countries in films')

train['production_countries'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_countries for i in j]).most_common(25)
train['num_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)

train['all_countries'] = train['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]

for g in top_countries:

    train['production_country_' + g] = train['all_countries'].apply(lambda x: 1 if g in x else 0)

    

test['num_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)

test['all_countries'] = test['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_countries:

    test['production_country_' + g] = test['all_countries'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['production_countries', 'all_countries'], axis=1)

test = test.drop(['production_countries', 'all_countries'], axis=1)
for i, e in enumerate(train['spoken_languages'][:5]):

    print(i, e)
print('Number of spoken languages in films')

train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0).value_counts()
list_of_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_languages for i in j]).most_common(15)
train['num_languages'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

train['all_languages'] = train['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]

for g in top_languages:

    train['language_' + g] = train['all_languages'].apply(lambda x: 1 if g in x else 0)

    

test['num_languages'] = test['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

test['all_languages'] = test['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_languages:

    test['language_' + g] = test['all_languages'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['spoken_languages', 'all_languages'], axis=1)

test = test.drop(['spoken_languages', 'all_languages'], axis=1)
for i, e in enumerate(train['Keywords'][:5]):

    print(i, e)
print('Number of Keywords in films')

train['Keywords'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)

train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]

for g in top_keywords:

    train['keyword_' + g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)

    

test['num_Keywords'] = test['Keywords'].apply(lambda x: len(x) if x != {} else 0)

test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_keywords:

    test['keyword_' + g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['Keywords', 'all_Keywords'], axis=1)

test = test.drop(['Keywords', 'all_Keywords'], axis=1)
for i, e in enumerate(train['cast'][:1]):

    print(i, e)
print('Number of casted persons in films')

train['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(15)
# 那些被抛弃的人对电影的质量有很大的影响。我们不仅有演员的名字，还有性别和角色的名字/类型。
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_names for i in j]).most_common(15)
list_of_cast_genders = list(train['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_genders for i in j]).most_common()
list_of_cast_characters = list(train['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_cast_characters for i in j]).most_common(15)
train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)

top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]

for g in top_cast_names:

    train['cast_name_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

train['genders_0_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['genders_1_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['genders_2_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]

for g in top_cast_characters:

    train['cast_character_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

    

test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)

for g in top_cast_names:

    test['cast_name_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)

test['genders_0_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['genders_1_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['genders_2_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

for g in top_cast_characters:

    test['cast_character_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)



train = train.drop(['cast'], axis=1)

test = test.drop(['cast'], axis=1)
for i, e in enumerate(train['crew'][:1]):

    print(i, e[:10])
print('Number of casted persons in films')

train['crew'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)
list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_names for i in j]).most_common(15)
list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)
list_of_crew_genders = list(train['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_genders for i in j]).most_common(15)
list_of_crew_departments = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)

Counter([i for j in list_of_crew_departments for i in j]).most_common(14)
train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)

top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]

for g in top_crew_names:

    train['crew_name_' + g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)

train['genders_0_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

train['genders_1_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

train['genders_2_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]

for g in top_cast_characters:

    train['crew_character_' + g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)

top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]

for j in top_crew_jobs:

    train['jobs_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))

top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]

for j in top_crew_departments:

    train['departments_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 

    

test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)

for g in top_crew_names:

    test['crew_name_' + g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)

test['genders_0_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

test['genders_1_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

test['genders_2_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

for g in top_cast_characters:

    test['crew_character_' + g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)

for j in top_crew_jobs:

    test['jobs_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))

for j in top_crew_departments:

    test['departments_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 



train = train.drop(['crew'], axis=1)

test = test.drop(['crew'], axis=1)
train.head()
test.head()
#收入分配

fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.hist(train['revenue']);

plt.title('Distribution of revenue');

plt.subplot(1, 2, 2)

plt.hist(np.log1p(train['revenue']));

plt.title('Distribution of log of revenue');
train['log_revenue'] = np.log1p(train['revenue'])
#收入和预测

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(train['budget'], train['revenue'])

plt.title('Revenue vs budget');

plt.subplot(1, 2, 2)

plt.scatter(np.log1p(train['budget']), train['log_revenue'])

plt.title('Log Revenue vs log budget');
#归一化

train['log_budget'] = np.log1p(train['budget'])

test['log_budget'] = np.log1p(test['budget'])
train['homepage'].value_counts().head()
#电影是否有无主页与收入的关系

train['has_homepage'] = 0

train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1

test['has_homepage'] = 0

test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1



sns.catplot(x='has_homepage', y='revenue', data=train);

plt.title('Revenue for film with and without homepage');
#电影语言与收入的关系

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

sns.boxplot(x='original_language', y='revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]);

plt.title('Mean revenue per language');

plt.subplot(1, 2, 2)

sns.boxplot(x='original_language', y='log_revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]);

plt.title('Mean log revenue per language');
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LinearRegression

import eli5
vectorizer = TfidfVectorizer(

            sublinear_tf=True,

            analyzer='word',

            token_pattern=r'\w{1,}',

            ngram_range=(1, 2),

            min_df=5)



overview_text = vectorizer.fit_transform(train['overview'].fillna(''))

linreg = LinearRegression()

linreg.fit(overview_text, train['log_revenue'])

eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')
print('Target value:', train['log_revenue'][1000])

eli5.show_prediction(linreg, doc=train['overview'].values[1000], vec=vectorizer)
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(train['popularity'], train['revenue'])

plt.title('Revenue vs popularity');

plt.subplot(1, 2, 2)

plt.scatter(train['popularity'], train['log_revenue'])

plt.title('Log Revenue vs popularity');
test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'
def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year
train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))

test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))

train['release_date'] = pd.to_datetime(train['release_date'])

test['release_date'] = pd.to_datetime(test['release_date'])
# creating features based on dates

def process_date(df):

    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        part_col = 'release_date' + "_" + part

        df[part_col] = getattr(df['release_date'].dt, part).astype(int)

    

    return df



train = process_date(train)

test = process_date(test)
import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.tools as tls
#每年的电影数量

d1 = train['release_date_year'].value_counts().sort_index()

d2 = test['release_date_year'].value_counts().sort_index()

data = [go.Scatter(x=d1.index, y=d1.values, name='train'), go.Scatter(x=d2.index, y=d2.values, name='test')]

layout = go.Layout(dict(title = "Number of films per year",

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Count'),

                  ),legend=dict(

                orientation="v"))

py.iplot(dict(data=data, layout=layout))
#每年的电影数量和总收入

d1 = train['release_date_year'].value_counts().sort_index()

d2 = train.groupby(['release_date_year'])['revenue'].sum()

data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), go.Scatter(x=d2.index, y=d2.values, name='total revenue', yaxis='y2')]

layout = go.Layout(dict(title = "Number of films and total revenue per year",

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Count'),

                  yaxis2=dict(title='Total revenue', overlaying='y', side='right')

                  ),legend=dict(

                orientation="v"))

py.iplot(dict(data=data, layout=layout))
#每年电影数量和平均收入

d1 = train['release_date_year'].value_counts().sort_index()

d2 = train.groupby(['release_date_year'])['revenue'].mean()

data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), go.Scatter(x=d2.index, y=d2.values, name='mean revenue', yaxis='y2')]

layout = go.Layout(dict(title = "Number of films and average revenue per year",

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Count'),

                  yaxis2=dict(title='Average revenue', overlaying='y', side='right')

                  ),legend=dict(

                orientation="v"))

py.iplot(dict(data=data, layout=layout))
# 我们可以看到电影的数量和总收入都在增长，

# 这是可以预料的。但在过去的几年里，成功的电影数量很多，带来了高额的收入。
sns.catplot(x='release_date_weekday', y='revenue', data=train);

plt.title('Revenue on different days of week of release');
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)

plt.hist(train['runtime'].fillna(0) / 60, bins=40);

plt.title('Distribution of length of film in hours');

plt.subplot(1, 3, 2)

plt.scatter(train['runtime'].fillna(0), train['revenue'])

plt.title('runtime vs revenue');

plt.subplot(1, 3, 3)

plt.scatter(train['runtime'].fillna(0), train['popularity'])

plt.title('runtime vs popularity');
# 看来大多数电影的片长都是1.5-2小时，收入最高的电影也在这个范围内
train['status'].value_counts()
test['status'].value_counts()
sns.catplot(x='num_genres', y='revenue', data=train);

plt.title('Revenue for different number of genres in the film');
sns.violinplot(x='genre_Drama', y='revenue', data=train[:100]);
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(train['num_crew'], train['revenue'])

plt.title('Number of crew members vs revenue');

plt.subplot(1, 2, 2)

plt.scatter(train['num_crew'], train['log_revenue'])

plt.title('Log Revenue vs number of crew members');
f, axes = plt.subplots(3, 5, figsize=(24, 18))

plt.suptitle('Violinplot of revenue vs jobs')

for i, e in enumerate([col for col in train.columns if 'jobs_' in col]):

    sns.violinplot(x=e, y='revenue', data=train, ax=axes[i // 5][i % 5]);
train = train.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue'], axis=1)

test = test.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status'], axis=1)
# for col in train.columns:

#     if train[col].nunique() == 1:

#         print(col)

#         train = train.drop([col], axis=1)

#         test = test.drop([col], axis=1)
for col in ['original_language', 'collection_name', 'all_genres']:

    le = LabelEncoder()

    le.fit(list(train[col].fillna('')) + list(test[col].fillna('')))

    train[col] = le.transform(train[col].fillna('').astype(str))

    test[col] = le.transform(test[col].fillna('').astype(str))
train_texts = train[['title', 'tagline', 'overview', 'original_title']]

test_texts = test[['title', 'tagline', 'overview', 'original_title']]
for col in ['title', 'tagline', 'overview', 'original_title']:

    train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))

    train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))

    train = train.drop(col, axis=1)

    test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))

    test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))

    test = test.drop(col, axis=1)
train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          

train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 

train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal

test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick

test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise

test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2

test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II

test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth

test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee



power_six = train.id[train.budget > 1000][train.revenue < 100]



for k in power_six :

    train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000
X = train.drop(['id', 'revenue','production_companies'], axis=1)

y = np.log1p(train['revenue'])

X_test = test.drop(['id','production_companies'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

# print(X_train.dtypes.value_counts())

# print(X_valid)

# print(y_train)

# print(y_valid)
lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# specify your configurations as a dict

params = {'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}

# train

print('Start training...')

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=10000,

                valid_sets=lgb_eval,

                early_stopping_rounds=100)



print('Start predicting...')



preds = gbm.predict(X_valid, num_iteration=gbm.best_iteration)  # 输出的是概率结果



# 导出结果

for pred in preds:

    result = prediction = int(np.argmax(pred))



# 导出特征重要性

# importance = gbm.feature_importance()

# names = gbm.feature_name()

# with open('./feature_importance.txt', 'w+') as file:

#     for index, im in enumerate(importance):

#         string = names[index] + ', ' + str(im) + '\n'

#         print(string)

#         file.write(string)

model1 = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

model1.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)
eli5.show_weights(model1, feature_filter=lambda x: x != '<BIAS>')
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):



    oof = np.zeros(X.shape[0])

    prediction = np.zeros(X_test.shape[0])

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print('Fold', fold_n, 'started at', time.ctime())

        if model_type == 'sklearn':

            X_train, X_valid = X[train_index], X[valid_index]

        else:

            X_train, X_valid = X.values[train_index], X.values[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

                    verbose=1000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test.values), ntree_limit=model.best_ntree_limit)



        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_squared_error(y_valid, y_pred_valid)

            

            y_pred = model.predict(X_test)

            

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric='RMSE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)

        

        prediction += y_pred    

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction

    

    else:

        return oof, prediction
params = {'num_leaves': 30,

         'min_data_in_leaf': 10,

         'objective': 'regression',

         'max_depth': 5,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}

oof_lgb, prediction_lgb, _ = train_model(X, X_test, y, params=params, model_type='lgb', plot_feature_importance=True)
for col in train_texts.columns:

    vectorizer = TfidfVectorizer(

                sublinear_tf=True,

                analyzer='word',

                token_pattern=r'\w{1,}',

                ngram_range=(1, 2),

                min_df=10

    )

    vectorizer.fit(list(train_texts[col].fillna('')) + list(test_texts[col].fillna('')))

    train_col_text = vectorizer.transform(train_texts[col].fillna(''))

    test_col_text = vectorizer.transform(test_texts[col].fillna(''))

    model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)

    oof_text, prediction_text = train_model(train_col_text, test_col_text, y, params=None, model_type='sklearn', model=model)

    

    X[col + '_oof'] = oof_text

    X_test[col + '_oof'] = prediction_text
def new_features(df):

    df['budget_to_popularity'] = df['budget'] / df['popularity']

    df['budget_to_runtime'] = df['budget'] / df['runtime']

    

    # some features from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3

    df['_budget_year_ratio'] = df['budget'] / (df['release_date_year'] * df['release_date_year'])

    df['_releaseYear_popularity_ratio'] = df['release_date_year'] / df['popularity']

    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_date_year']

    

    df['runtime_to_mean_year'] = df['runtime'] / df.groupby("release_date_year")["runtime"].transform('mean')

    df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_date_year")["popularity"].transform('mean')

    df['budget_to_mean_year'] = df['budget'] / df.groupby("release_date_year")["budget"].transform('mean')

        

    return df
X = new_features(X)

X_test = new_features(X_test)
oof_lgb, prediction_lgb, _ = train_model(X, X_test, y, params=params, model_type='lgb', plot_feature_importance=True)
xgb_params = {'eta': 0.01,

              'objective': 'reg:linear',

              'max_depth': 7,

              'subsample': 0.8,

              'colsample_bytree': 0.8,

              'eval_metric': 'rmse',

              'seed': 11,

              'silent': True}

oof_xgb, prediction_xgb= train_model(X, X_test, y, params=xgb_params, model_type='xgb', plot_feature_importance=True)
cat_params = {'learning_rate': 0.002,

              'depth': 5,

              'l2_leaf_reg': 10,

              # 'bootstrap_type': 'Bernoulli',

              'colsample_bylevel': 0.8,

              'bagging_temperature': 0.2,

              #'metric_period': 500,

              'od_type': 'Iter',

              'od_wait': 100,

              'random_seed': 11,

              'allow_writing_files': False}

oof_cat, prediction_cat = train_model(X, X_test, y, params=cat_params, model_type='cat')
train_stack = np.vstack([oof_lgb, oof_xgb, oof_cat]).transpose()

train_stack = pd.DataFrame(train_stack, columns=['lgb', 'xgb', 'cat'])

test_stack = np.vstack([prediction_lgb, prediction_xgb, prediction_cat]).transpose()

test_stack = pd.DataFrame(test_stack, columns=['lgb', 'xgb', 'cat'])
arams = {'num_leaves': 8,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 2,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

         "verbosity": -1}

oof_lgb_stack, prediction_lgb_stack, _ = train_model(train_stack, test_stack, y, params=params, model_type='lgb', plot_feature_importance=True)
model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)

oof_rcv_stack, prediction_rcv_stack = train_model(train_stack.values, test_stack.values, y, params=None, model_type='sklearn', model=model)
sub = pd.read_csv('../input/sample_submission.csv')

sub['revenue'] = np.expm1(prediction_lgb)

sub.to_csv("lgb.csv", index=False)

sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb) / 2)

sub.to_csv("blend.csv", index=False)

sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb + prediction_cat) / 3)

sub.to_csv("blend1.csv", index=False)

sub['revenue'] = prediction_lgb_stack

sub.to_csv("stack_lgb.csv", index=False)

sub['revenue'] = prediction_rcv_stack

sub.to_csv("stack_rcv.csv", index=False)