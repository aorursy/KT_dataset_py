import numpy as np 

import pandas as pd

import json

import ast

from collections import Counter, OrderedDict

import time

import datetime

import random

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from sklearn.model_selection import train_test_split, KFold

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

import eli5



import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

df_test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
features=df_train.drop(['revenue'],axis=1).append(df_test).reset_index()
# from this kernel: https://www.kaggle.com/gravix/gradient-in-a-box

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
def text_to_dict(df, columns_to_parse):

    for column in columns_to_parse:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df
f_clean = text_to_dict(features,dict_columns)
def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year
f_clean.loc[f_clean['release_date'].isnull() == True, 'release_date'] = '01/01/98' 
f_clean['release_date'] = f_clean['release_date'].apply(lambda x: fix_date(x))

f_clean['release_date'] = pd.to_datetime(f_clean['release_date'])
f_clean['year']=pd.DatetimeIndex(f_clean['release_date']).year

f_clean['month']=pd.DatetimeIndex(f_clean['release_date']).month

f_clean['yr_mth']=f_clean['year']*100+f_clean['month']
min_date_months = f_clean["year"].min()*12 + f_clean["month"].min()



def change_time_to_num(year_month, min_date):

    date_to_months = year_month.apply(lambda x: int(str(x)[:4]) * 12 + int(str(x)[-2:]))

    return date_to_months.apply(lambda x: x - min_date)



f_clean['timediff'] = change_time_to_num(f_clean['yr_mth'], min_date_months)
f_clean['original_title'][f_clean.duplicated('original_title')].shape
f_clean['edited_title'] = f_clean['original_title'].copy()

f_clean['edited_title'][f_clean.duplicated('original_title')] = f_clean['edited_title'].map(str) + ' (' + f_clean['year'].map(str) + ')'
f_clean['edited_title'][f_clean.duplicated('edited_title')].shape
movie_index={v: k for k, v in f_clean['edited_title'].to_dict().items()}

index_movie=f_clean['edited_title'].to_dict()
f_clean['list_keywords']=f_clean['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
len(set([i for j in f_clean['list_keywords'] for i in j])) # check number of unique keywords
count_keywords=Counter([i for j in f_clean['list_keywords'] for i in j]).most_common()
count_keywords[0:10] # check most common keywords
for i in range(1,5):

    print(f'There are {len([t[0] for t in count_keywords if t[1] == i])} keywords that appear in {i} movies')



print(f'There are {len([t[0] for t in count_keywords if t[1] > 4])} keywords that appear in 5 or more movies')

print(f'In total, {len([t[0] for t in count_keywords if t[1] > 1])} keywords appear more than once')
keywords = [t[0] for t in count_keywords if t[1] > 1]
kcount=pd.concat([f_clean['edited_title'],

                  f_clean['list_keywords'].apply(lambda x: [i for i in x if i in keywords]),

                  f_clean['list_keywords'].apply(lambda x: len([i for i in x if i in keywords]))],

                 axis=1)

kcount.columns=['movie','keywords','kcount']
kcount.sort_values(by='kcount',ascending=False)[0:10]
kcount['movie'].loc[kcount['kcount']==0].count() # check how many movies have zero keywords
kword_index = {kword: idx for idx, kword in enumerate(keywords)}

index_kword = {idx: kword for kword, idx in kword_index.items()}
pairs = []



for movie in movie_index.values():

    pairs.extend((movie,kword_index[kword]) for kword in kcount['keywords'][kcount.index==movie].iloc[0]) 
random.seed(100)



pairs_set = set(pairs)



def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):

    """Generate batches of samples for training"""

    batch_size = n_positive * (1 + negative_ratio)

    batch = np.zeros((batch_size, 3))

    

    # Adjust label based on task

    if classification:

        neg_label = 0

    else:

        neg_label = -1

    

    # This creates a generator

    while True:

        # randomly choose positive examples

        for idx, (movie_id, kword_id) in enumerate(random.sample(pairs, n_positive)):

            batch[idx, :] = (movie_id, kword_id, 1)



        # Increment idx by 1

        idx += 1

        

        # Add negative examples until reach batch size

        while idx < batch_size:

            

            # random selection

            random_movie = random.randrange(len(index_movie))

            random_kword = random.randrange(len(index_kword))

            

            # Check to make sure this is not a positive example

            if (random_movie, random_kword) not in pairs_set:

                

                # Add to batch and increment index

                batch[idx, :] = (random_movie, random_kword, neg_label)

                idx += 1

                

        # Make sure to shuffle order

        np.random.shuffle(batch)

        yield {'movie': batch[:, 0], 'kword': batch[:, 1]}, batch[:, 2]
from keras.layers import Input, Embedding, Dot, Reshape, Dense

from keras.models import Model


def embedding_model(embedding_size = 50, classification = False):

      

    # Layer 1: 1-dimensional inputs

    movie = Input(name = 'movie', shape = [1])

    kword = Input(name = 'kword', shape = [1])

    

    # Layer 2: Embedding the movie (shape will be (None, 1, 50))

    movie_embedding = Embedding(name = 'movie_embedding',

                               input_dim = len(movie_index),

                               output_dim = embedding_size)(movie)

    

    # Layer 2: Embedding the keyword (shape will be (None, 1, 50))

    kword_embedding = Embedding(name = 'kword_embedding',

                               input_dim = len(kword_index),

                               output_dim = embedding_size)(kword)

    

    # Layer 3: Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))

    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([movie_embedding, kword_embedding])

    

    # Layer 4: Reshape to be a single number (shape will be (None, 1))

    merged = Reshape(target_shape = [1])(merged)

    

    # If classifcation, add extra layer and loss function is binary cross entropy

    if classification:

        merged = Dense(1, activation = 'sigmoid')(merged) # layer 5: for classification

        model = Model(inputs = [movie, kword], outputs = merged)

        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    

    # Otherwise loss function is mean squared error

    else:

        model = Model(inputs = [movie, kword], outputs = merged)

        model.compile(optimizer = 'Adam', loss = 'mse')

    

    return model
model = embedding_model()

model.summary()
n_positive = 1024



gen = generate_batch(pairs, n_positive, negative_ratio = 2)



# Train

h = model.fit_generator(gen, epochs = 50, 

                        steps_per_epoch = len(pairs) // n_positive,

                        verbose = 2)
movie_layer = model.get_layer('movie_embedding')

movie_weights = movie_layer.get_weights()[0]

movie_weights.shape
movie_weights = movie_weights / np.linalg.norm(movie_weights, axis = 1).reshape((-1, 1))
def similar_movies(name, n=10):

    

    dists = np.dot(movie_weights, movie_weights[movie_index[name]])

    sorted_dists = np.argsort(dists)

    closest = sorted_dists[-n:]

    max_width = max([len(index_movie[c]) for c in closest])

    

    for c in reversed(closest):

        print(f'Movie: {index_movie[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')
similar_movies('Avatar')
col_names = ['membed_'+str(i) for i in range(1,51)]



movie_embeds = pd.DataFrame(movie_weights,columns=col_names)
f_clean = pd.concat([f_clean,movie_embeds],axis=1)
train = pd.concat([f_clean.iloc[0:df_train.shape[0]],df_train['revenue']],axis=1)

test = f_clean.iloc[df_train.shape[0]:]
train['revenue'].describe().T
train[['original_title','revenue']][train['revenue']<300].sort_values(by=['revenue'])
train['revenue'][train['revenue']<300] = 300
train['director'] = train['crew'].apply(lambda x: [i['name'] for i in x if i['job'] == 'Director']).apply(pd.Series).iloc[:,0]
train['revenue'].groupby(train['director']).count().describe().T
dir_rev= pd.concat([train['revenue'].groupby(train['director']).count(),

           train['revenue'].groupby(train['director']).sum(),

           train['revenue'].groupby(train['director']).mean(),

           train['revenue'].groupby(train['director']).max(),

           train['revenue'].groupby(train['director']).min()], axis=1).reset_index()



dir_rev.columns = ['director','N_movies','Total_rev','Average_rev','Highest_rev','Lowest_rev']
dir_rev[dir_rev['N_movies']>1].sort_values(by=['Average_rev'],ascending=False)[0:10]
dir_rev['Hi_lo_rev'] = (dir_rev['Highest_rev'] - dir_rev['Lowest_rev']) / dir_rev['Average_rev']
f_clean['director'] = f_clean['crew'].apply(lambda x: [i['name'] for i in x if i['job'] == 'Director']).apply(pd.Series).iloc[:,0]
f_clean = f_clean.merge(dir_rev[['director','Average_rev','Hi_lo_rev']],how='left',on='director')

f_clean = f_clean.rename(columns={'Average_rev': 'Dir_avg_rev', 'Hi_lo_rev': 'Dir_HL_rev'})
cast_list = train['cast'].apply(lambda x: [i['name'] for i in x])
len(set([i for j in cast_list for i in j])) # over 38,000 unique cast members in the training set
cast_revenue = []



for i,r in enumerate(train['revenue'].values):

    cast_revenue.extend((act,r) for act in cast_list[cast_list.index==i].iloc[0])

    

cast_revenue = pd.DataFrame(list(cast_revenue), columns=['Name','Revenue'])



cast_rev_sum = pd.concat([cast_revenue.groupby(['Name']).count(),

                          cast_revenue.groupby(['Name']).sum(),

                          cast_revenue.groupby(['Name']).mean(),

                          cast_revenue.groupby(['Name']).max(),

                          cast_revenue.groupby(['Name']).min()], axis=1)



cast_rev_sum.columns = ['N_movies','Total_rev','Average_rev','Highest_rev','Lowest_rev']
cast_rev_sum.sort_values(by=['Highest_rev'],ascending=False)[0:10]
cast_rev_sum['rev99p'] = (cast_rev_sum['Highest_rev'] >cast_revenue['Revenue'].quantile(0.99))*1

cast_rev_sum['rev20p'] = (cast_rev_sum['Highest_rev'] <cast_revenue['Revenue'].quantile(0.20))*1
full_cast_list = f_clean['cast'].apply(lambda x: [i['name'] for i in x])
id_cast = []



for i,r in enumerate(f_clean['id'].values):

    id_cast.extend((r,act) for act in full_cast_list[full_cast_list.index==i].iloc[0])

    

id_cast = pd.DataFrame(list(id_cast), columns=['id','Name'])
cast_rev_movie = id_cast.merge(cast_rev_sum,how='left',on='Name')
cast_rev_summary = pd.concat([cast_rev_movie.groupby(['id']).sum()['rev99p'],

                             cast_rev_movie.groupby(['id']).sum()['rev20p'],

                             cast_rev_movie.groupby(['id']).min()['Highest_rev']],

                             axis=1).reset_index()
f_clean = f_clean.merge(cast_rev_summary,how='left',on='id')

f_clean = f_clean.rename(columns={'rev99p': 'N_cast_99p', 'rev20p': 'N_cast_20p','Highest_rev':'Cast_low_bound'})
companies_list = train['production_companies'].apply(lambda x: [i['name'] for i in x])
comp_revenue = []



for i,r in enumerate(train['revenue'].values):

    comp_revenue.extend((comp,r) for comp in companies_list[companies_list.index==i].iloc[0])

    

comp_revenue = pd.DataFrame(list(comp_revenue), columns=['Company','Revenue'])



comp_rev_sum = pd.concat([comp_revenue.groupby(['Company']).count(),

                          comp_revenue.groupby(['Company']).sum(),

                          comp_revenue.groupby(['Company']).mean(),

                          comp_revenue.groupby(['Company']).max(),

                          comp_revenue.groupby(['Company']).min()], axis=1)



comp_rev_sum.columns = ['N_movies','Total_rev','Average_rev','Highest_rev','Lowest_rev']
comp_rev_sum['rev75p'] = (comp_rev_sum['Highest_rev'] >comp_revenue['Revenue'].quantile(0.75))*1

comp_rev_sum['rev25p'] = (comp_rev_sum['Highest_rev'] <comp_revenue['Revenue'].quantile(0.25))*1
full_comp_list = f_clean['production_companies'].apply(lambda x: [i['name'] for i in x])
id_comp = []



for i,r in enumerate(f_clean['id'].values):

    id_comp.extend((r,comp) for comp in full_comp_list[full_comp_list.index==i].iloc[0])

    

id_comp = pd.DataFrame(list(id_comp), columns=['id','Company'])
comp_rev_movie = id_comp.merge(comp_rev_sum,how='left',on='Company')
comp_rev_summary = pd.concat([comp_rev_movie.groupby(['id']).sum()['rev75p'],

                             comp_rev_movie.groupby(['id']).sum()['rev25p'],

                             comp_rev_movie.groupby(['id']).min()['Highest_rev']],

                             axis=1).reset_index()
f_clean = f_clean.merge(comp_rev_summary,how='left',on='id')

f_clean = f_clean.rename(columns={'rev75p': 'N_comp_75p', 'rev25p': 'N_comp_25p','Highest_rev':'Comp_low_bound'})
med_budget = train['budget'].median()
f_clean['budget'] = f_clean['budget'].replace(0, med_budget)
f_clean['runtime'] = f_clean['runtime'].fillna(train['runtime'].median())

f_clean['Dir_avg_rev'] = f_clean['Dir_avg_rev'].fillna(f_clean['Dir_avg_rev'].median())

f_clean['Dir_HL_rev'] = f_clean['Dir_HL_rev'].fillna(f_clean['Dir_HL_rev'].median())

f_clean['N_cast_99p'] = f_clean['N_cast_99p'].fillna(0)

f_clean['N_cast_20p'] = f_clean['N_cast_20p'].fillna(0)

f_clean['Cast_low_bound'] = f_clean['Cast_low_bound'].fillna(f_clean['Cast_low_bound'].median())

f_clean['N_comp_75p'] = f_clean['N_comp_75p'].fillna(1)

f_clean['N_comp_25p'] = f_clean['N_comp_25p'].fillna(0)

f_clean['Comp_low_bound'] = f_clean['Comp_low_bound'].fillna(f_clean['Comp_low_bound'].median())

extra_train = pd.read_csv('../input/moviestmdb-datapreparation/train_prep.csv')

extra_test = pd.read_csv('../input/moviestmdb-datapreparation/test_prep.csv')
cols_add = ['has_collection','num_cast','num_crew','genres_name_Drama','genres_name_Comedy','genres_name_Thriller',

            'genres_name_Action','genres_name_Romance','genres_name_Crime','genres_name_Adventure',

            'genres_name_Horror','genres_name_Science Fiction','genres_name_Family',

            'production_countries_name_United States of America',

            'spoken_languages_name_English','spoken_languages_name_Français','spoken_languages_name_Español']
train = pd.concat([f_clean.iloc[0:df_train.shape[0]],extra_train[cols_add],df_train['revenue']],axis=1)

test = pd.concat([f_clean.iloc[df_train.shape[0]:].reset_index(),extra_test[cols_add]],axis=1)
cols_to_drop = ['index','id','belongs_to_collection','genres','homepage','imdb_id','original_language',

                'original_title','overview','poster_path','production_companies','production_countries',

               'release_date','spoken_languages','status','tagline','title','Keywords','cast','crew',

               'edited_title','list_keywords','director']
X = train.drop(['revenue'],axis=1).drop(cols_to_drop,axis=1)

y = np.log1p(train['revenue'])

X_test = test.drop(['level_0'],axis=1).drop(cols_to_drop,axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
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



lgbm = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

lgbm.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)
eli5.show_weights(lgbm, feature_filter=lambda x: x != '<BIAS>')
lasso = linear_model.Lasso(alpha=0.1)

print(np.sqrt(-cross_val_score(lasso, X, y, cv=10, scoring='neg_mean_squared_error')))
lasso.fit(X,y)
preds_lasso = lasso.predict(X_test)
sub = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')

sub['revenue'] = np.expm1(preds_lasso)

sub.to_csv("lasso_sub.csv", index=False)