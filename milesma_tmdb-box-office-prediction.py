# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import seaborn as sns

from sklearn import *

import xgboost as xg

# import GPy

# import GPyOpt



# from GPyOpt.methods import BayesianOptimization



# Any results you write to the current directory are saved as output.
# Libraries



import numpy as np

import pandas as pd

from functools import reduce

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

import json

import ast

import math

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

from sklearn.metrics import mean_squared_log_error
or_train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

or_test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')



# from this kernel: https://www.kaggle.com/gravix/gradient-in-a-box

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

or_train = text_to_dict(or_train)

or_test = text_to_dict(or_test)



train = or_train.copy()

test = or_test.copy()
train['belongs_to_collection'].head()
chs = ['belongs_to_collection']

for table in [train, test]:

    for col in chs:

        trs = table[col].apply(lambda x:sorted(list(map(lambda y:y['name'], x))))

        text = trs.apply(lambda x:" ".join(x))

        table[col] = text
text_train = or_train.copy()

text_test = or_test.copy()



text_train=text_train.fillna('')

text_test=text_test.fillna('')



text_cols = ['belongs_to_collection','genres','production_companies','production_countries','Keywords','cast','crew']

for col in text_cols:

    trs=text_train[col].apply(lambda x:sorted(list(map(lambda y:y['name'], x))))

    text = trs.apply(lambda x:" ".join(x)+" " if len(x)>0 else "")

    text_train[col] = text.apply(lambda x:x.strip())

    

    trs=text_test[col].apply(lambda x:sorted(list(map(lambda y:y['name'], x))))

    text = trs.apply(lambda x:" ".join(x)+" " if len(x)>0 else "")

    text_test[col] = text.apply(lambda x:x.strip())
choose_text = ['belongs_to_collection','genres','production_companies','production_countries','overview']

text_train['text'] = text_train[choose_text].apply(lambda x: ''.join(x), axis=1)

text_train = text_train['text']

text_train = text_train.apply(lambda x:x.split(' '))



text_test['text'] = text_test[choose_text].apply(lambda x: ''.join(x), axis=1)

text_test = text_test['text']

text_test = text_test.apply(lambda x:x.split(' '))
embeddings_index = {}

with open(os.path.join('../input/glove6b50dtxt/glove.6B.50d.txt')) as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs
EMBEDDING_DIM = max(max(text_train.apply(lambda x:len(x))), max(text_test.apply(lambda x:len(x))))*50

num_words = len(text_train)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

test_matrix = np.zeros((len(text_test), EMBEDDING_DIM))
for row in range(len(text_train)):

    line = text_train[row]

    for i in range(len(line)):

        k = line[i].lower()

        if k in embeddings_index:

            vec = embeddings_index[k]

            embedding_matrix[row][i*50:(i+1)*50] = vec

for row in range(len(text_test)):

    line = text_test[row]

    for i in range(len(line)):

        k = line[i].lower()

        if k in embeddings_index:

            vec = embeddings_index[k]

            test_matrix[row][i*50:(i+1)*50] = vec
rand_indices = np.random.permutation(3000)

train_indices = rand_indices[0:2500]

valid_indices = rand_indices[2500:]



x_val = embedding_matrix[valid_indices]

# y_val = ys[valid_indices]



x_tr = embedding_matrix[train_indices]

# y_tr = ys[train_indices]



print('Shape of x_tr: ' + str(x_tr.shape))

# print('Shape of y_tr: ' + str(y_tr.shape))

print('Shape of x_val: ' + str(x_val.shape))

# print('Shape of y_val: ' + str(y_val.shape))
from keras.layers import Dense, Input

from keras import models

from keras import regularizers

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization



input_img = Input(shape=(EMBEDDING_DIM,), name='input_img')



encode1 = Dense(EMBEDDING_DIM//2, activation='relu', name='encode1')(input_img)

hidden = Dropout(0.5)(encode1)

encode2 = Dense(128, activation='relu', name='encode2')(hidden)

encode3 = Dense(32, activation='relu', name='encode3')(encode2)

bottleneck = Dense(8, activation='relu', name='bottleneck')(encode3)

decode1 = Dense(32, activation='relu', name='decode1')(bottleneck)

decode2 = Dense(128, activation='relu', name='decode2')(decode1)

decode3 = Dense(EMBEDDING_DIM//2, activation='relu', name='decode3')(decode2)

hidden = Dropout(0.5)(decode3)

decode4 = Dense(EMBEDDING_DIM, activation='relu', name='decode4')(decode3)



# hidden1 = Dense(128, activation='relu')(bottleneck)

# hidden2 = Dense(265, activation='relu', kernel_regularizer=regularizers.l2(0.01))(hidden1)

# hidden3 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(hidden2)

# hidden4 = Dropout(0.5)(hidden3)

# hidden5 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(hidden4)

# classifier = Dense(1, activation='linear', name='reg', activity_regularizer=regularizers.l1(0.01))(hidden5)



ae = models.Model(input_img, decode4)
from keras import optimizers



learning_rate = 1E-3 # to be tuned!



ae.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=learning_rate))

history = ae.fit(x_tr, x_tr, 

                 batch_size=128, 

                 epochs=20, 

                 validation_data=(x_val, x_val))
import matplotlib.pyplot as plt

%matplotlib inline



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(loss))



plt.plot(epochs, loss, 'bo', label='Training Loss')

plt.plot(epochs, val_loss, 'r', label='Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
ae_encoder = models.Model(input_img, bottleneck)

encoded_test = ae_encoder.predict(test_matrix)

encoded_test

encoded_train = ae_encoder.predict(embedding_matrix)

encoded_train
xg_train = xg.DMatrix(data=train.loc[:,['budget','popularity','runtime']], label=np.array(train['revenue']))

params = {'eval_metric' : 'rmse', 'silent' : 1}

xg_model = xg.train(params, xg_train)

xg_test = xg.DMatrix(data=train.loc[:,['budget', 'popularity', 'runtime']])

xg_pred = xg_model.predict(xg_test)



eli5.show_weights(xg_model)
fig, ax = plt.subplots(figsize=(20, 10))

plt.scatter(train['budget'], train['revenue'])



linereg = linear_model.LinearRegression()

linereg.fit(np.array(train['budget']).reshape(-1, 1),train['revenue'])

li_pre = linereg.predict(np.array(train['budget']).reshape(-1, 1))

plt.plot(train['budget'], li_pre, color='orange', label='linear')



plt.scatter(train['budget'], xg_pred, label = 'xgboost')

plt.legend()

plt.show()
print(np.sqrt(metrics.mean_squared_log_error(train['revenue'], li_pre)))

print(np.sqrt(metrics.mean_squared_log_error(train['revenue'], xg_pred)))
train['log_budget'] = np.log1p(train['budget'])

train['log_revenue'] = np.log1p(train['revenue'])

test['log_budget'] = np.log1p(test['budget'])
sns.boxenplot(x='original_language', y='log_revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)])
for table in [train, test]:

    table['has_homepage'] = 0

    table.loc[table['homepage'].isnull() == True, 'has_homepage'] = 0

    table.loc[table['homepage'].isnull() == False, 'has_homepage'] = 1
sns.catplot(x='has_homepage', y='revenue', data=train)
plt.figure(figsize=(16,8))

plt.subplot(1,2,1)

sns.boxenplot(x='original_language', y='revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)])

plt.subplot(1,2,2)

sns.boxenplot(x='original_language', y='log_revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)])
# plt.figure(figsize=(12,12))

# text = " ".join(train.loc[train['revenue'].sort_values(ascending=False).head(200).index, 'crew'])

# wc = WordCloud(background_color='white', width=1200, height=1000).generate(text)

# plt.axis("off")

# plt.imshow(wc)
# vec = TfidfVectorizer(sublinear_tf=True, analyzer='word', ngram_range=(1,2), min_df=5)

# overview_text = vec.fit_transform(train['overview'].fillna(''))

# linreg = LinearRegression()

# linreg.fit(overview_text, train['log_revenue'])

# eli5.show_weights(linreg, vec=vec, top=20)
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)

plt.hist(train['runtime'].fillna(0), bins=40)

plt.subplot(1,2,2)

plt.scatter(train['runtime'].fillna(0), train['revenue'])
def fix_date(x):

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2]+"20"+year

    else:

        return x[:-2]+"19"+year



test.loc[test['release_date'].isnull(), 'release_date'] = '5/1/00'

for table in [train, test]:

    table['release_date'] = table['release_date'].apply(lambda x:fix_date(x))
def trans(table):

    df = pd.DataFrame()

    t = pd.to_datetime(table['release_date'])

    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        part_col = 'release_date' + "_" + part

        df[part_col] = getattr(t.dt, part).astype(int)

    table = pd.concat([table, df], axis=1)

    return table

train = trans(train)

test = trans(test)
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
sns.catplot(x='release_date_year', y='revenue', data=train);

train.head()
train['num_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)

test['num_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)

sns.catplot(x='num_countries', y='revenue', data=train);

plt.title('Revenue for different number of countries producing the film');
train = train.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status','log_revenue'], axis=1)

test = test.drop(['homepage', 'imdb_id', 'poster_path', 'release_date','status'], axis=1)
train.head()
train.rename(columns={'belongs_to_collection':"collection_name"}, inplace=True)

test.rename(columns={'belongs_to_collection':"collection_name"}, inplace=True)
train['log_budget'] = np.log1p(train['budget'])

test['log_budget'] = np.log1p(test['budget'])
list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

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
list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)

train['all_production_companies'] = train['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]

for g in top_companies:

    train['production_company_' + g] = train['all_production_companies'].apply(lambda x: 1 if g in x else 0)

    

test['num_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)

test['all_production_companies'] = test['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

for g in top_companies:

    test['production_company_' + g] = test['all_production_companies'].apply(lambda x: 1 if g in x else 0)



train = train.drop(['production_companies', 'all_production_companies'], axis=1)

test = test.drop(['production_companies', 'all_production_companies'], axis=1)
list_of_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

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
list_of_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

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
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

list_of_cast_genders = list(train['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

list_of_cast_characters = list(train['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)

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
list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)

list_of_crew_genders = list(train['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)

list_of_crew_departments = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)

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
for table in [test, train]:

    table['has_collection'] = table['collection_name'].apply(lambda x: 1 if len(x)!=0 else 0)
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
X = train.drop(['id', 'revenue'], axis=1)

y = np.log1p(train['revenue'])

X_test = test.drop(['id'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
print(X_train.shape)

print(X_valid.shape)

print(y_train.shape)

print(y_valid.shape)

print(X_test.shape)
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
y.shape
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
# xg_train = xg.DMatrix(data=X, label=np.array(y))

# params = {'eval_metric' : 'rmse', 'silent' : 1}

# xg_model = xg.train(params, xg_train)

# xg_test = xg.DMatrix(data=X_test)

# xg_pred = xg_model.predict(xg_test)
df2 = pd.DataFrame(encoded_train)

X=pd.concat([X, df2], axis=1)



df2 = pd.DataFrame(encoded_test)

X_test = pd.concat([X_test, df2], axis=1)



X
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBRegressor

xg_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

               colsample_bytree=1, gamma=4.541003990662603, importance_type='gain',

               learning_rate=0.08209238500752991, max_delta_step=0, max_depth=4,

               min_child_weight=8, missing=None, n_estimators=137, n_jobs=1,

               nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,

               reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,

               subsample=1)



xg_train = xg.DMatrix(data=X, label=np.array(y))

xg_model.fit(X, y)

xg_test = xg.DMatrix(data=X_test)

xg_pred = xg_model.predict(X_test)
train_pred = xg_model.predict(X)

train_pred = np.expm1(train_pred)

np.sqrt(mean_squared_log_error(train['revenue'], train_pred))
sub = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')

sub['revenue'] = np.expm1(xg_pred)

sub.to_csv("xgb.csv", index=False)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBRegressor



# xgb1 = XGBRegressor()

# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

#               'objective':['reg:linear'],

#               'learning_rate': [.03, 0.05, .07], #so called `eta` value

#               'max_depth': [5, 6, 7],

#               'min_child_weight': [4],

#               'silent': [1],

#               'subsample': [0.7],

#               'colsample_bytree': [0.7],

#               'n_estimators': [500]}





# folds = 3

# param_comb = 5



# kf = KFold(n_splits=10)



# random_search = RandomizedSearchCV(xgb, param_distributions=parameters, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=kf.split(X,y), verbose=3, random_state=1001 )

# random_search.fit(X, y)



# print(xgb_grid.best_score_)

# print(xgb_grid.best_params_)
from sklearn import datasets

from sklearn.model_selection import RandomizedSearchCV, cross_val_score



from scipy.stats import uniform

from xgboost import XGBRegressor



xgb = XGBRegressor()



baseline = cross_val_score(xgb, X, y, scoring='neg_mean_squared_error').mean()

baseline
param_dist = {"learning_rate": uniform(0, 1),

              "gamma": uniform(0, 5),

              "max_depth": range(1,50),

              "n_estimators": range(1,300),

              "min_child_weight": range(1,10)}



rs = RandomizedSearchCV(xgb, param_distributions=param_dist, 

                        scoring='neg_mean_squared_error', n_iter=25)



# Run random search for 25 iterations

rs.fit(X, y);
# random_search = rs

# print('\n All results:')

# print(random_search.cv_results_)

# print('\n Best estimator:')

# print(random_search.best_estimator_)

# import GPy

# import GPyOpt

# from GPyOpt.methods import BayesianOptimization



# bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},

#         {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},

#         {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},

#         {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},

#         {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]



# # Optimization objective 

# def cv_score(parameters):

#     parameters = parameters[0]

#     score = cross_val_score(

#                 XGBRegressor(learning_rate=parameters[0],

#                               gamma=int(parameters[1]),

#                               max_depth=int(parameters[2]),

#                               n_estimators=int(parameters[3]),

#                               min_child_weight = parameters[4]), 

#                 X, y, scoring='neg_mean_squared_error').mean()

#     score = np.array(score)

#     return score



# optimizer = BayesianOptimization(f=cv_score, 

#                                  domain=bds,

#                                  model_type='GP',

#                                  acquisition_type ='EI',

#                                  acquisition_jitter = 0.05,

#                                  exact_feval=True, 

#                                  maximize=True)



# # Only 20 iterations because we have 5 initial random points

# optimizer.run_optimization(max_iter=20)
# optimizer.Y_best
y_rs = np.maximum.accumulate(rs.cv_results_['mean_test_score'])

# y_bo = np.maximum.accumulate(-optimizer.Y).ravel()



print(f'Baseline neg. MSE = {baseline:.2f}')

print(f'Random search neg. MSE = {y_rs[-1]:.2f}')

# print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')



plt.plot(y_rs, 'ro-', label='Random search')

# plt.plot(y_bo, 'bo-', label='Bayesian optimization')

plt.xlabel('Iteration')

plt.ylabel('Neg. MSE')

plt.title('Value of the best sampled CV score');

plt.legend();