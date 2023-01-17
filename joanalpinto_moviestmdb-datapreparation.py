# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import ast



# Any results you write to the current directory are saved as output.



import json

import ast

from collections import Counter

import time

import datetime

import os
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# from this kernel: https://www.kaggle.com/gravix/gradient-in-a-box

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
def text_to_dict(df, columns_to_parse):

    for column in columns_to_parse:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df
df_train_clean = text_to_dict(df_train,dict_columns)

df_test_clean = text_to_dict(df_test, dict_columns)
df_train_clean['has_collection'] = df_train_clean['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

df_test_clean['has_collection'] = df_test_clean['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
columns_to_select = ['genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
def number_of_values(df, columns):

    for column in columns:

        new_column_name = 'num_' + column

        df[new_column_name] = df[column].apply(lambda x: len(x) if x != {} else 0)

    return df
df_train_clean = number_of_values(df_train_clean, columns_to_select)

df_test_clean = number_of_values(df_test_clean, columns_to_select)
def list_of_values(df, columns, key):

    for column in columns:

        new_column_name = 'all_' + column + '_' + key

        df[new_column_name] = df[column].apply(lambda x: ' '.join(sorted([i[key] for i in x])) if x != {} else '')

    return df
columns_to_select = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
df_train_clean = list_of_values(df_train_clean, columns_to_select, 'name')

df_test_clean = list_of_values(df_test_clean, columns_to_select, 'name')
columns_top_to_select = {'belongs_to_collection':32, 'genres': 10, 'production_companies': 20,

                'production_countries': 15, 'spoken_languages': 15, 'Keywords':30, 'cast':30, 'crew':20}
def top_values(df_calculate_top, df_apply_top, columns_to_select, key):

    for column, top_value in columns_to_select.items():

        list_of_values = df_calculate_top[column].apply(lambda x: [i[key] for i in x] if x != {} else []).values

        top_values = [m[0] for m in Counter([i for j in list_of_values for i in j]).most_common(top_value)]

        for value in top_values:

            df_apply_top[column+'_'+key+'_'+value] = df_apply_top['all_'+column+'_'+key].apply(lambda x: 1 if value in x else 0)

    return df_apply_top
df_train_clean = top_values(df_train_clean, df_train_clean, columns_top_to_select, 'name')

df_test_clean = top_values(df_train_clean, df_test_clean, columns_top_to_select, 'name')
df_train_clean = list_of_values(df_train_clean, ['cast'], 'character')

df_test_clean = list_of_values(df_test_clean, ['cast'], 'character')



columns_top_to_select = {'cast': 10}

df_train_clean = top_values(df_train_clean, df_train_clean, columns_top_to_select, 'character')

df_test_clean = top_values(df_train_clean, df_test_clean, columns_top_to_select, 'character')
#count gender

dict_gender = {0:'genunk', 1:'female', 2:'male'}
def count_gender(df, dict_gender, column, key):

    for k, v in dict_gender.items():

        df['num' + '_' + column + '_' + key + '_' + v] = df[column].apply(lambda x: sum([1 for i in x if i[key] == k]))

    return df
#cast gender

df_train_clean = count_gender(df_train_clean, dict_gender, 'cast', 'gender')

df_test_clean = count_gender(df_test_clean, dict_gender, 'cast', 'gender')
#crew gender

df_train_clean = count_gender(df_train_clean, dict_gender, 'crew', 'gender')

df_test_clean = count_gender(df_test_clean, dict_gender, 'crew', 'gender')
def top_jobs(df, all_values, column, new_column_name, top):

    top = [m[0] for m in Counter([i for j in all_values for i in j]).most_common(top)]

    for value in top:

        df[new_column_name+'_'+value] = df[column].apply(lambda x: 1 if value in str(x) else 0)

    return df
list_directors = list(df_train_clean['crew'].apply(lambda x: [i['name'] for i in x if i['job'] == 'Director'] if x != {} else []).values)

df_train_clean = top_jobs(df_train_clean, list_directors, 'crew', 'director', 30)

df_test_clean = top_jobs(df_test_clean, list_directors, 'crew', 'director', 30)
list_producers = list(df_train_clean['crew'].apply(lambda x: [i['name'] for i in x if i['job'] == 'Executive Producer'] if x != {} else []).values)

df_train_clean = top_jobs(df_train_clean, list_producers, 'crew', 'producer', 15)

df_test_clean = top_jobs(df_test_clean, list_producers, 'crew', 'producer', 15)
def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year
df_test_clean.loc[df_test_clean['release_date'].isnull() == True, 'release_date'] = '01/01/98' 
df_train_clean['release_date'] = df_train_clean['release_date'].apply(lambda x: fix_date(x))

df_test_clean['release_date'] = df_test_clean['release_date'].apply(lambda x: fix_date(x))

df_train_clean['release_date'] = pd.to_datetime(df_train_clean['release_date'])

df_test_clean['release_date'] = pd.to_datetime(df_test_clean['release_date'])
df_train_clean['year']=pd.DatetimeIndex(df_train_clean['release_date']).year

df_test_clean['year']=pd.DatetimeIndex(df_test_clean['release_date']).year

df_train_clean['month']=pd.DatetimeIndex(df_train_clean['release_date']).month

df_test_clean['month']=pd.DatetimeIndex(df_test_clean['release_date']).month

df_train_clean['yr_mth']=df_train_clean['year']*100+df_train_clean['month']

df_test_clean['yr_mth']=df_test_clean['year']*100+df_test_clean['month']
def find_features_start_by(df, start):

    features = list(df)

    start_by = []

    for feature in features:

        if feature.startswith(start):

            start_by.append(feature)

    return start_by

        
#drop from train set

features_to_drop = find_features_start_by(df_train_clean, 'all_')

df_train_clean = df_train_clean.drop(features_to_drop, axis=1)



#drop from test set

df_test_clean = df_test_clean.drop(features_to_drop, axis=1)

original_features = ['belongs_to_collection', 'genres', 'production_companies', 'production_countries',

                     'spoken_languages', 'Keywords', 'cast', 'crew', 'release_date']
df_train_clean = df_train_clean.drop(original_features, axis=1)

df_test_clean = df_test_clean.drop(original_features, axis=1)
print('training set: ' + str(len(list(df_train_clean))) + ' features')

print('testing set: ' + str(len(list(df_test_clean))) + ' features')
i = 1

for column in list(df_train_clean.columns):

    print(str(i) + ' ' + column)

    i = i + 1
df_train_clean.to_csv('../working/train_prep.csv', index=False)

df_test_clean.to_csv('../working/test_prep.csv', index=False)