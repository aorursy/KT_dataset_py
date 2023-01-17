# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')

train.info()
test = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')

test.info()
# import Libraries

import seaborn as sns

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

import plotly as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from collections import Counter
train.head(5)
train.shape
test.shape
train.columns
train.isna().values.sum()
# Get Name From Data

def get_name(data):

    try:

        x = eval(data) 

        return x[0]['name']

    except:

        return ''



# Get length of data

def get_len(data):

    try:

        x = eval(data)

        return len(x)

    except:

        return 0



# Get all data by column

def all_data(data, key = 'name'):

    try:

        x = eval(data)

        return ' '.join(sorted([i[key] for i in x]))

    except:

        return 



# all data of column convert into list 

def make_list(data, key = 'name'):

    try:

        data = eval(data)

        return [i[key] for i in data]

    except: 

        return []



# applying one hot encoding for multiple data

def apply_encode(data, name):

    try:

        if name in data:

            return 1

        else:

            return 0

    except:

        return



# Get Gender Type [0,1,2]  

def get_gender(data,index):

    try:

        data = eval(str(data))

        count = 0

        for i in data:

            if i['gender'] == index:

                count += 1

        return count

    except:

        return 0



# Change feature name to singular verb

def singular_verb(feature_name):

    if feature_name == 'production_countries':

        return 'production_country'

    elif feature_name == 'production_companies':

        return 'production_company'

    elif feature_name == 'spoken_languages':

        return 'spoken_language'

    elif feature_name == 'Keywords':

        return 'Keyword'

    else:

        return feature_name



# Fixes dates which are in 20xx

def fix_date(x):

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year



# creating features based on dates

def process_date(df):

    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        part_col = 'release_date' + "_" + part

        df[part_col] = getattr(df['release_date'].dt, part).astype(int)

    return df



#...............................

#

# Remove same data

# def unique_list(data):

#    if data:

#        k=set()

#        for i in data:

#            for m in i:

#                k.add(m)

#        return list(k)

#

# old count word function

# count = 0

# def count_word(data, name):

#    try:

#        if name in data:

#            global count

#            count += 1

#            return name

#    except:

#        return

#

# old comman word finder function

# def comman_word(feature, all_word, fetch = 25):

#    make_count = []

#    for name in all_word:

#        global count

#        count = 0

#       not_to_print = feature.apply(lambda x:count_word(x, name))

#        make_count.append([name, count])

#        print([name, count])

#    k = sorted(make_count, key = lambda x: x[1])[::-1]

#   return np.array(k)[:fetch,0]

#

#.................................
# train missing values

train['runtime'].fillna(0, inplace=True)

train['status'].fillna('Released', inplace=True)

train['release_date'].fillna(train['release_date'].mode()[0], inplace=True)



# test missing values

test['runtime'].fillna(0, inplace=True)

test['status'].fillna('Released', inplace=True)

test['release_date'].fillna(test['release_date'].mode()[0], inplace=True)
# train data cleaning

train['collection_name'] = train['belongs_to_collection'].apply(get_name)

train['num_of_collection'] = train['belongs_to_collection'].apply(get_len)

train['num_of_genres'] = train['genres'].apply(get_len)

train['num_of_countries'] = train['production_countries'].apply(get_len)

train['num_of_companies'] = train['production_companies'].apply(get_len)

train['num_of_spoken_languages'] = train['spoken_languages'].apply(get_len)

train['num_of_cast'] = train['cast'].apply(get_len)

train['num_of_crew'] = train['crew'].apply(get_len)

train['num_of_keywords'] = train['Keywords'].apply(get_len)

train['has_homepage'] = 0

train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1



# test data cleaning

test['collection_name'] = test['belongs_to_collection'].apply(get_name)

test['num_of_collection'] = test['belongs_to_collection'].apply(get_len)

test['num_of_genres'] = test['genres'].apply(get_len)

test['num_of_countries'] = test['production_countries'].apply(get_len)

test['num_of_companies'] = test['production_companies'].apply(get_len)

test['num_of_spoken_languages'] = test['spoken_languages'].apply(get_len)

test['num_of_cast'] = test['cast'].apply(get_len)

test['num_of_crew'] = test['crew'].apply(get_len)

test['num_of_keywords'] = test['Keywords'].apply(get_len)

test['has_homepage'] = 0

test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1
# feature name

feature_list = ['genres', 'production_countries', 'production_companies',

                'spoken_languages', 'Keywords']

# train feature encoding

for feature_name in feature_list: 

    field = train[feature_name].apply(all_data)

    list_feature = train[feature_name].apply(make_list)

    list_of_feature = list(list_feature.values)

    top_feature_data = [c[0] for c in Counter([i for j in list_of_feature for i in j]).most_common(12)]

    feature_name_sv = singular_verb(feature_name)

    for name in top_feature_data:

        train[feature_name_sv + '_' + name] = field.apply(lambda x:apply_encode(x,name))

        train[feature_name_sv + '_' + name] = train[feature_name_sv + '_' + name].fillna(0).astype('int32')

    train = train.drop([feature_name], axis=1)



# test feature encoding

for feature_name in feature_list: 

    field = test[feature_name].apply(all_data)

    list_feature = test[feature_name].apply(make_list)

    list_of_feature = list(list_feature.values)

    top_feature_data = [c[0] for c in Counter([i for j in list_of_feature for i in j]).most_common(12)]

    feature_name_sv = singular_verb(feature_name)

    for name in top_feature_data:

        test[feature_name_sv + '_' + name] = field.apply(lambda x:apply_encode(x,name))

        test[feature_name_sv + '_' + name] = test[feature_name_sv + '_' + name].fillna(0).astype('int32')

    test = test.drop([feature_name], axis=1)
# cast and crew encoding train

feature_list = ['cast','crew']

for feature_name in feature_list:

    if feature_name == 'cast':

        # cast_keys

        list_keys = ['name','character']

        for i_key in list_keys:

            feature = train[feature_name].apply(lambda x: all_data(x,i_key))

            list_feature = train[feature_name].apply(lambda x: make_list(x,i_key))

            list_of_feature = list(list_feature.values)

            top_feature_data = [c[0] for c in Counter([i for j in list_of_feature for i in j]).most_common(10)]

            feature_name_sv = singular_verb(feature_name)

            for name in top_feature_data:

                train[feature_name_sv  + '_'+ i_key + '_' + name] = feature.apply(lambda x:apply_encode(x,name))

                train[feature_name_sv  + '_'+ i_key + '_' + name] = train[feature_name_sv  + '_'+ i_key + '_' + name].fillna(0).astype('int32')

            # feature and list_feature are dataframe so it can't be reassign.

            del feature

            del list_feature

    if feature_name == 'crew':

        # crew_keys

        list_keys = ['name','job','department']

        for i_key in list_keys:

            feature = train[feature_name].apply(lambda x: all_data(x,i_key))

            list_feature = train[feature_name].apply(lambda x: make_list(x,i_key))

            list_of_feature = list(list_feature.values)

            top_feature_data = [c[0] for c in Counter([i for j in list_of_feature for i in j]).most_common(10)]

            feature_name_sv = singular_verb(feature_name)

            for name in top_feature_data:

                train[feature_name_sv + '_'+ i_key + '_' + name] = feature.apply(lambda x:apply_encode(x,name))

                train[feature_name_sv + '_'+ i_key + '_' + name] = train[feature_name_sv + '_'+ i_key + '_' + name].fillna(0).astype('int32')

            # feature and list_feature are dataframe so it can't be reassign. 

            del feature

            del list_feature

    # cast and crew gender encoding

    train[feature_name +'_genders_0'] = train[feature_name].apply(lambda x: get_gender(x,0))

    train[feature_name +'_genders_1'] = train[feature_name].apply(lambda x: get_gender(x,1))

    train[feature_name +'_genders_2'] = train[feature_name].apply(lambda x: get_gender(x,2))



# cast and crew encoding test

feature_list = ['cast','crew']

for feature_name in feature_list:

    if feature_name == 'cast':

        # cast_keys

        list_keys = ['name','character']

        for i_key in list_keys:

            feature = test[feature_name].apply(lambda x: all_data(x,i_key))

            list_feature = test[feature_name].apply(lambda x: make_list(x,i_key))

            list_of_feature = list(list_feature.values)

            top_feature_data = [c[0] for c in Counter([i for j in list_of_feature for i in j]).most_common(10)]

            feature_name_sv = singular_verb(feature_name)

            for name in top_feature_data:

                test[feature_name_sv  + '_'+ i_key + '_' + name] = feature.apply(lambda x:apply_encode(x,name))

                test[feature_name_sv  + '_'+ i_key + '_' + name] = test[feature_name_sv  + '_'+ i_key + '_' + name].fillna(0).astype('int32')

            # feature and list_feature are dataframe so it can't be reassign.

            del feature

            del list_feature

    if feature_name == 'crew':

        # crew_keys

        list_keys = ['name','job','department']

        for i_key in list_keys:

            feature = test[feature_name].apply(lambda x: all_data(x,i_key))

            list_feature = test[feature_name].apply(lambda x: make_list(x,i_key))

            list_of_feature = list(list_feature.values)

            top_feature_data = [c[0] for c in Counter([i for j in list_of_feature for i in j]).most_common(10)]

            feature_name_sv = singular_verb(feature_name)

            for name in top_feature_data:

                test[feature_name_sv + '_'+ i_key + '_' + name] = feature.apply(lambda x:apply_encode(x,name))

                test[feature_name_sv + '_'+ i_key + '_' + name] = test[feature_name_sv + '_'+ i_key + '_' + name].fillna(0).astype('int32')

            # feature and list_feature are dataframe so it can't be reassign. 

            del feature

            del list_feature

    # cast and crew gender encoding

    test[feature_name +'_genders_0'] = test[feature_name].apply(lambda x: get_gender(x,0))

    test[feature_name +'_genders_1'] = test[feature_name].apply(lambda x: get_gender(x,1))

    test[feature_name +'_genders_2'] = test[feature_name].apply(lambda x: get_gender(x,2))
# train release_date feature process

train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))

train['release_date'] = pd.to_datetime(train['release_date'])

train = process_date(train)



# test release_date feature process

test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))

test['release_date'] = pd.to_datetime(test['release_date'])

test = process_date(test)
for col in ['original_language', 'collection_name']:

    le = LabelEncoder()

    le.fit(list(train[col].fillna('')) + list(test[col].fillna('')))

    train[col] = le.transform(train[col].fillna('').astype(str))

    test[col] = le.transform(test[col].fillna('').astype(str))
# drop columns

train = train.drop(['belongs_to_collection','cast','crew','homepage', 'imdb_id', 'poster_path', 'status'], axis=1)

test = test.drop(['belongs_to_collection','cast','crew','homepage', 'imdb_id', 'poster_path', 'status'], axis=1)
d1 = train['release_date_year'].value_counts().sort_index()

d2 = train.groupby(['release_date_year'])['revenue'].sum()

data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), 

        go.Scatter(x=d2.index, y=d2.values, name='total revenue', yaxis='y2')]

layout = go.Layout(dict(title = "Number of films and total revenue per year",

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Count'),

                  yaxis2=dict(title='Total revenue', overlaying='y', side='right')

                  ),legend=dict(

                orientation="v"))

py.offline.iplot(dict(data=data, layout=layout))
d1 = train['release_date_year'].value_counts().sort_index()

d2 = train.groupby(['release_date_year'])['revenue'].mean()

data = [go.Scatter(x=d1.index, y=d1.values, name='film count'), go.Scatter(x=d2.index, y=d2.values, name='mean revenue', yaxis='y2')]

layout = go.Layout(dict(title = "Number of films and average revenue per year",

                  xaxis = dict(title = 'Year'),

                  yaxis = dict(title = 'Count'),

                  yaxis2=dict(title='Average revenue', overlaying='y', side='right')

                  ),legend=dict(

                orientation="v"))

py.offline.iplot(dict(data=data, layout=layout))
sns.catplot(x='release_date_weekday', y='revenue', data=train);

plt.title('Revenue on different days of week of release');
plt.figure(figsize=(20, 6))

plt.subplot(1, 3, 1)

plt.hist(train['runtime']/ 60, bins=40);

plt.title('Distribution of length of film in hours');

plt.subplot(1, 3, 2)

plt.scatter(train['runtime'], train['revenue'])

plt.title('runtime vs revenue');

plt.subplot(1, 3, 3)

plt.scatter(train['runtime'], train['popularity'])

plt.title('runtime vs popularity');
sns.catplot(x='num_of_genres', y='revenue', data=train);

plt.title('Revenue for different number of genres in the film');
sns.catplot(x='num_of_countries', y='revenue', data=train);

plt.title('Revenue for different number of countries producing the film');
# drop column with only 1 value

for col in train.columns:

    if train[col].nunique() == 1:

        print(col)

        train = train.drop([col], axis=1)

        test = test.drop([col], axis=1)
train=train.drop(['original_title', 'overview','release_date','tagline','title'], axis=1)

test = test.drop(['original_title', 'overview','release_date','tagline','title'], axis=1)
# Check Null

train.isnull().any().sum()
x_train = train.drop(['id', 'revenue'], axis=1).values

y_train = np.log1p(train['revenue']).values

test_id = test['id']

test = test.drop(['id'], axis=1).values
xg_reg = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.001,

                max_depth = 3, alpha = 10, n_estimators = 15000)
xg_reg.fit(x_train,y_train)
test_pred = xg_reg.predict(test)

test_pred = np.expm1(test_pred)
# dictionary of lists  

dict = {'id': test_id, 'revenue': test_pred}  

     

df = pd.DataFrame(dict) 

  

# saving the dataframe 

df.to_csv('my_submission_file.csv',index=False) 