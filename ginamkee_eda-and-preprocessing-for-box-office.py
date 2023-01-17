import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.graph_objs as go

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')
train.head()
print("Dataset contains {} rows and {} columns".format(train.shape[0], train.shape[1]))
print("Dataset contains {} rows and {} columns".format(test.shape[0], test.shape[1]))
train.info()
for col in train.columns:

         print('{}\n'.format(train[col].head()))
train['popularity'].describe()
corrMatrix=train[['revenue', 'cast', 'runtime', 'production_countries', 'budget', 'popularity', 'release_date', 'title', 'genres', 'original_language']].corr()

sns.set(font_scale=1.10)

plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,

square=True,annot=True,cmap='viridis',linecolor="white")

plt.title('Correlation between features');
corr_matrix = train.corr()

corr_matrix["revenue"].sort_values(ascending=False)
train['logRevenue'] = np.log1p(train['revenue'])

sns.distplot(train['logRevenue'])
train[['release_month','release_day','release_year']]=train['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)



train.loc[ (train['release_year'] <= 19) & (train['release_year'] < 100), "release_year"] += 2000

train.loc[ (train['release_year'] > 19)  & (train['release_year'] < 100), "release_year"] += 1900
releaseDate = pd.to_datetime(train['release_date']) 

train['release_dayofweek'] = releaseDate.dt.day_name()

train['release_quarter'] = releaseDate.dt.quarter
plt.figure(figsize=(15,10))

sns.countplot(x='release_dayofweek', data=train)
plt.figure(figsize=(18,12))

plt.xticks(fontsize=12,rotation=90)

sns.countplot(x='release_year', data=train)
sns.countplot(x='release_quarter', data=train)
train.info()
train['revenue'].describe()
sns.scatterplot(x='budget', y='revenue', data=train, color = 'g')
sns.scatterplot(x='popularity', y='revenue', data=train, color = 'r')
train['production_companies'].value_counts()[:11]
train['production_companies_count'] = train.groupby('production_companies')['production_companies'].transform('count')

test['production_companies_count'] = test.groupby('production_companies')['production_companies'].transform('count')
top_11_companies = train.loc[(train['production_companies_count'] >= 12)]
def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace

cnt_srs = top_11_companies.groupby('production_companies')['revenue'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace0 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(100, 71, 96, 0.6)')

layout = go.Layout(title = '', width=1000, height=700)

fig = go.Figure(data = trace0, layout = layout)

fig
sns.scatterplot(x='production_companies_count', y='revenue', data=train, color = 'r')
sns.scatterplot(x='runtime', y='revenue', data=train)
train.loc[(train['status'] == 'Rumored')]['release_date']
train.replace('Rumored', 'Released', inplace=True)
train['status'].value_counts()
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
train.loc[1819,'genres']='Ramance, Drama'

train.loc[470,'genres']='Adventure, Drama'

train.loc[1622,'genres']='Drama, Comedy'

train.loc[1814,'genres']='Comedy'

train.loc[2423,'genres']='Action'

train.loc[2686,'genres']='Thriller'

train.loc[2900,'genres']='Drama'
train.loc[1335, 'runtime'] = 130

train.loc[2302, 'runtime'] = 110
train['poster_path'].fillna(method='backfill')

test['poster_path'].fillna(method='backfill')
train['poster_path'] = train['poster_path'].fillna(method='backfill')
train[train['poster_path'].isnull()]
total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
lan = train['spoken_languages'].mode()[0]

train['spoken_languages'] = train['spoken_languages'].fillna(lan)



lant = test['spoken_languages'].mode()[0]

test['spoken_languages'] = test['spoken_languages'].fillna(lant)
com = train['production_companies'].mode()[0]

train['production_companies'] = train['production_companies'].fillna(com)



comt = test['production_companies'].mode()[0]

test['production_companies'] = test['production_companies'].fillna(comt)
con = train['production_countries'].mode()[0]

train['production_countries'] = train['production_countries'].fillna(con)



cont = test['production_countries'].mode()[0]

test['production_countries'] = test['production_countries'].fillna(cont)
key = train['Keywords'].mode()[0]

train['Keywords'] = train['Keywords'].fillna(key)



keyt = test['Keywords'].mode()[0]

test['Keywords'] = test['Keywords'].fillna(keyt)
gen = test['genres'].mode()[0]

test['genres'] = test['genres'].fillna(gen)
train['tagline'].fillna(0, inplace=True)

train['crew'].fillna(0, inplace=True)

train['cast'].fillna(0, inplace=True)

train['overview'].fillna(0, inplace=True)

train['production_companies_count'].fillna(1,inplace=True)



test['overview'].fillna(0, inplace=True)

test['tagline'].fillna(0, inplace=True)

test['crew'].fillna(0, inplace=True)

test['cast'].fillna(0, inplace=True)

test['production_companies_count'].fillna(1,inplace=True)
test['runtime'].fillna((test['runtime'].mean()), inplace=True)
train = train.drop(['belongs_to_collection','homepage','status'], axis=1)

test = test.drop(['belongs_to_collection','homepage','status'], axis=1)
train['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()
for i, e in enumerate(train['production_companies'][:5]):

    print(i, e)
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
test.isnull().sum()
test.loc[2398,'title'] = 'グスコーブドリの伝記'

test.loc[2425, 'title'] = 'La Vérité si je Mens ! 3'

test.loc[3628, 'title'] = 'Barefoot'
test.loc[828, 'release_date'] = 2000
test.loc[828, 'poster_path'] = '/baz1c9dzsf5uhNuUYhXy7eudNJd.jpg'
train['popularity'].median()
train.loc[(train['popularity'] > 200)][['title','popularity']]