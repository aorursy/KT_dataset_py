# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from collections import OrderedDict



import ast

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/train.csv')

train_df.head()

#train_df[train_df['popularity'] > float(80)]

y = train_df['revenue']

train_df = train_df.drop(['belongs_to_collection', 'homepage'], axis=1)
def sep_dict(s,key):

    '''

    Separate row Dict by key ('id' or 'name')

    '''

    l = ast.literal_eval(s)

    l = [str(x[key]) for x in l]

    if key == 'id':

        return ' '.join(l)

    else:

        return l



def encode_column(s, colname):

    '''

    Encode the row value using 1 to length of items in row

    '''

    legend = build_legend(colname)

    l = s.split(' ')

    l = [str(legend[int(x)]) for x in l if x != '']

    return l



def build_legend(colname):

    '''

    Builds the legend used to encode the 

    columns in encode_column()

    '''

    ids = list(train_df[colname].unique())

    unique_ids = []

    unique_ids = sorted(list(set([int(z) for x in ids for z in x.split(' ') if z not in unique_ids and z != ''])))

    id_legend = dict(zip(unique_ids, range(1, len(unique_ids)+1)))

    return id_legend



#Run each of the previous helper functions over specific rows containing dicts

for colname in ['production_companies', 'cast', 'Keywords', 'genres']:

    col_names = f'{colname}_names'

    col_ids = f'{colname}_ids'

    train_df[colname] = train_df[colname].fillna('[]')

    train_df[col_names] = train_df[colname].apply(lambda x: sep_dict(x, 'name'))

    train_df[col_ids] = train_df[colname].apply(lambda x: sep_dict(x, 'id'))

    train_df[col_ids] = train_df[col_ids].apply(lambda x: encode_column(x, col_ids))



#Focusing on English origin movies

en_train_df = train_df[train_df['original_language'] == 'en']
en_train_df
def get_sequels(title):

    pattern = re.compile(r'\s[2-9](?:$|\:|\s)')

    search = re.search(pattern,title)

    if re.search(pattern,title) == None:

        return 0

    else:

        return search.group(0)[1]



en_train_df['sequels'] = en_train_df.title.apply(lambda x: get_sequels(x))
#en_train_df[en_train_df['popularity'] > 100]

en_train_df.columns
fig, ax = plt.subplots(figsize=(10,10))

sns.distplot(en_train_df.popularity)
sns.scatterplot(x=en_train_df.popularity, y=y)
drop_cols = ['id', 'imdb_id', 'original_language', 'overview', 'poster_path', 'production_companies',

            'status', 'original_title', 'Keywords', 'cast', 'crew', 'production_companies', 'spoken_languages',

            'production_companies']

id_drop_cols = drop_cols + ['production_companies_names', 'cast_names', 'Keywords_names', 'genres_names']



n_id_df = en_train_df.drop(drop_cols, axis=1)

id_df = en_train_df.drop(id_drop_cols, axis=1)
n_id_df.head()
def get_uniques(df, colname):

    raw_list = []

    for x in df[colname]:

        for actor in x:

            raw_list.append(actor)

    return raw_list



cast_raw_ids = get_uniques(id_df, 'cast_ids')

cast_raw_names = get_uniques(n_id_df, 'cast_names')
cast_legend = dict(zip(cast_raw_ids, cast_raw_names))
cast_ids_counts = {cid:cast_raw_ids.count(cid) for cid in set(cast_raw_ids)}



cast_ids_counts = {x : cast_ids_counts[x] for x in sorted(cast_ids_counts, 

                                                          key=lambda x : cast_ids_counts[x], 

                                                          reverse=True)}

cast_ids_counts
unique_cast_counts = []

cast_counts_list = [x for x in cast_ids_counts.values()]

unique_cast_set = set(cast_counts_list)

unique_cast_counts = {x:cast_counts_list.count(x) for x in unique_cast_set}

unique_cast_counts
total = sum(x for x in unique_cast_counts.values()) - 25415

total
legend_dict