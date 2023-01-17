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
trainAdditionalFeatures = pd.read_csv('../input/tmdb-competition-additional-features/TrainAdditionalFeatures.csv')

testAdditionalFeatures = pd.read_csv('../input/tmdb-competition-additional-features/TestAdditionalFeatures.csv')



train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')



train = pd.merge(train, trainAdditionalFeatures, how='left', on=['imdb_id'])

test = pd.merge(test, testAdditionalFeatures, how='left', on=['imdb_id'])

test['revenue'] = -np.inf

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

train.loc[train['id'] == 1007,'budget'] = 2              # Zyzzyx Road 

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 1885,'budget'] = 12             # In the Cut

train.loc[train['id'] == 2091,'budget'] = 10             # Deadfall

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'budget'] = 6              # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

train.loc[train['id'] == 335,'budget'] = 2 

train.loc[train['id'] == 348,'budget'] = 12

train.loc[train['id'] == 470,'budget'] = 13000000 

train.loc[train['id'] == 513,'budget'] = 1100000

train.loc[train['id'] == 640,'budget'] = 6 

train.loc[train['id'] == 696,'budget'] = 1

train.loc[train['id'] == 797,'budget'] = 8000000 

train.loc[train['id'] == 850,'budget'] = 1500000

train.loc[train['id'] == 1199,'budget'] = 5 

train.loc[train['id'] == 1282,'budget'] = 9               # Death at a Funeral

train.loc[train['id'] == 1347,'budget'] = 1

train.loc[train['id'] == 1755,'budget'] = 2

train.loc[train['id'] == 1801,'budget'] = 5

train.loc[train['id'] == 1918,'budget'] = 592 

train.loc[train['id'] == 2033,'budget'] = 4

train.loc[train['id'] == 2118,'budget'] = 344 

train.loc[train['id'] == 2252,'budget'] = 130

train.loc[train['id'] == 2256,'budget'] = 1 

train.loc[train['id'] == 2696,'budget'] = 10000000



#Clean Data

test.loc[test['id'] == 6733,'budget'] = 5000000

test.loc[test['id'] == 3889,'budget'] = 15000000

test.loc[test['id'] == 6683,'budget'] = 50000000

test.loc[test['id'] == 5704,'budget'] = 4300000

test.loc[test['id'] == 6109,'budget'] = 281756

test.loc[test['id'] == 7242,'budget'] = 10000000

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee

test.loc[test['id'] == 3033,'budget'] = 250 

test.loc[test['id'] == 3051,'budget'] = 50

test.loc[test['id'] == 3084,'budget'] = 337

test.loc[test['id'] == 3224,'budget'] = 4  

test.loc[test['id'] == 3594,'budget'] = 25  

test.loc[test['id'] == 3619,'budget'] = 500  

test.loc[test['id'] == 3831,'budget'] = 3  

test.loc[test['id'] == 3935,'budget'] = 500  

test.loc[test['id'] == 4049,'budget'] = 995946 

test.loc[test['id'] == 4424,'budget'] = 3  

test.loc[test['id'] == 4460,'budget'] = 8  

test.loc[test['id'] == 4555,'budget'] = 1200000 

test.loc[test['id'] == 4624,'budget'] = 30 

test.loc[test['id'] == 4645,'budget'] = 500 

test.loc[test['id'] == 4709,'budget'] = 450 

test.loc[test['id'] == 4839,'budget'] = 7

test.loc[test['id'] == 3125,'budget'] = 25 

test.loc[test['id'] == 3142,'budget'] = 1

test.loc[test['id'] == 3201,'budget'] = 450

test.loc[test['id'] == 3222,'budget'] = 6

test.loc[test['id'] == 3545,'budget'] = 38

test.loc[test['id'] == 3670,'budget'] = 18

test.loc[test['id'] == 3792,'budget'] = 19

test.loc[test['id'] == 3881,'budget'] = 7

test.loc[test['id'] == 3969,'budget'] = 400

test.loc[test['id'] == 4196,'budget'] = 6

test.loc[test['id'] == 4221,'budget'] = 11

test.loc[test['id'] == 4222,'budget'] = 500

test.loc[test['id'] == 4285,'budget'] = 11

test.loc[test['id'] == 4319,'budget'] = 1

test.loc[test['id'] == 4639,'budget'] = 10

test.loc[test['id'] == 4719,'budget'] = 45

test.loc[test['id'] == 4822,'budget'] = 22

test.loc[test['id'] == 4829,'budget'] = 20

test.loc[test['id'] == 4969,'budget'] = 20

test.loc[test['id'] == 5021,'budget'] = 40 

test.loc[test['id'] == 5035,'budget'] = 1 

test.loc[test['id'] == 5063,'budget'] = 14 

test.loc[test['id'] == 5119,'budget'] = 2 

test.loc[test['id'] == 5214,'budget'] = 30 

test.loc[test['id'] == 5221,'budget'] = 50 

test.loc[test['id'] == 4903,'budget'] = 15

test.loc[test['id'] == 4983,'budget'] = 3

test.loc[test['id'] == 5102,'budget'] = 28

test.loc[test['id'] == 5217,'budget'] = 75

test.loc[test['id'] == 5224,'budget'] = 3 

test.loc[test['id'] == 5469,'budget'] = 20 

test.loc[test['id'] == 5840,'budget'] = 1 

test.loc[test['id'] == 5960,'budget'] = 30

test.loc[test['id'] == 6506,'budget'] = 11 

test.loc[test['id'] == 6553,'budget'] = 280

test.loc[test['id'] == 6561,'budget'] = 7

test.loc[test['id'] == 6582,'budget'] = 218

test.loc[test['id'] == 6638,'budget'] = 5

test.loc[test['id'] == 6749,'budget'] = 8 

test.loc[test['id'] == 6759,'budget'] = 50 

test.loc[test['id'] == 6856,'budget'] = 10

test.loc[test['id'] == 6858,'budget'] =  100

test.loc[test['id'] == 6876,'budget'] =  250

test.loc[test['id'] == 6972,'budget'] = 1

test.loc[test['id'] == 7079,'budget'] = 8000000

test.loc[test['id'] == 7150,'budget'] = 118

test.loc[test['id'] == 6506,'budget'] = 118

test.loc[test['id'] == 7225,'budget'] = 6

test.loc[test['id'] == 7231,'budget'] = 85

test.loc[test['id'] == 5222,'budget'] = 5

test.loc[test['id'] == 5322,'budget'] = 90

test.loc[test['id'] == 5350,'budget'] = 70

test.loc[test['id'] == 5378,'budget'] = 10

test.loc[test['id'] == 5545,'budget'] = 80

test.loc[test['id'] == 5810,'budget'] = 8

test.loc[test['id'] == 5926,'budget'] = 300

test.loc[test['id'] == 5927,'budget'] = 4

test.loc[test['id'] == 5986,'budget'] = 1

test.loc[test['id'] == 6053,'budget'] = 20

test.loc[test['id'] == 6104,'budget'] = 1

test.loc[test['id'] == 6130,'budget'] = 30

test.loc[test['id'] == 6301,'budget'] = 150

test.loc[test['id'] == 6276,'budget'] = 100

test.loc[test['id'] == 6473,'budget'] = 100

test.loc[test['id'] == 6842,'budget'] = 30



# from this kernel: https://www.kaggle.com/gravix/gradient-in-a-box

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)

test = text_to_dict(test)



def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    if not isinstance(x, str): return x

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year



train.loc[train['release_date'].isnull() == True, 'release_date'] = '01/01/19'

test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/19'

    

#train["RevByBud"] = train["revenue"] / train["budget"]

    

train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))

test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))

# save set of both datasets

datasets = (train, test)
# work with log revenue

train['lrevenue'] = np.log1p(train['revenue'])

for df in datasets:

    df['lbudget'] = np.log1p(df['budget'])
# mean reveneu feature

def plot_over_time(feature='revenue', split=None):

    sns.set(font_scale = 1)

    f, ax = plt.subplots(1, 1, figsize=(10,10))

    

    if split is None:

        sns.lineplot(x="year", y=feature, data=train, ax=ax)

        sns.regplot(x="year", y=feature, data=train, ax=ax, scatter=False)

    else:

        sns.lineplot(x="year", y=feature, hue=split, data=train, ax=ax)

#         sns.lmplot(x="year", y=feature, hue=split, data=train, scatter=False, palette="muted")
def plot_reveneu_over_feature(feature):

    sns.set(font_scale = 2)

    f, axes = plt.subplots(1, 3, figsize=(45,15))

    

#     targets = []

#     lables = []



    # Sort the dataframe by target 

    all_values = sorted(train[feature].unique())

    targets = [train.loc[train[feature]==all_values[i]]['lrevenue'] for i in range(len(all_values))]

    lables = all_values



    for lbl, dist in zip(lables, targets):

#         sns.lineplot(dist, hist=False, rug=False, label=lbl)

        sns.distplot(dist, hist=False, rug=True, label=lbl, ax=axes[0])

    # violin

#     sns.violinplot(x=feature, y="revenue", data=train, ax=axes[1])

    sns.barplot(x=feature, y="revenue", data=train, ax=axes[1])

    # pie

    # Plot

    size = train[feature].value_counts().sort_values()

    axes[2].pie(list(size), labels=size.index, autopct='%1.1f%%', shadow=True, startangle=140)

    

#     plt.axis('equal')

#     plt.show()
def plot_scatter(y='revenue', x='budget', order=1):

    sns.set(font_scale = 1)

    f, ax = plt.subplots(1, 1, figsize=(10,5))

    sns.regplot(x=x, y=y, data=train, x_estimator=np.mean, order=order)
def is_weelend(day):

    w_e = ['Friday', 'Sunday', 'Saturday']

    return 1 if day in w_e else 0

for df in datasets:

    df.release_date = pd.to_datetime(df.release_date)

    df.release_date = pd.to_datetime(df.release_date)



    df['day_in_week'] = df['release_date'].dt.day_name()

    df['day_in_month'] = df['release_date'].apply(lambda d: d.day)

    df['month'] = df['release_date'].apply(lambda d: d.month)

    df['year'] = df['release_date'].apply(lambda d: d.year)



    df['in_weekend'] = df.day_in_week.apply(is_weelend)
# convert features to amount

features_to_num = ['production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

for f in features_to_num:

    for df in datasets:

        df['num_'+f] = df[f].apply(len)
train.revenue.hist()
train.lrevenue.hist()
features_over_time = ['lrevenue', 'lbudget', 'popularity', 'runtime', 'popularity2', 'rating',

                     'totalVotes', 'num_production_companies', 'num_production_countries',

                     'num_spoken_languages', 'num_Keywords', 'num_cast']

for f in features_over_time:

    plot_over_time(feature=f)

    
# change belongs_to_collection to a binary feature (true or false)

for df in datasets:

    df['from_collection'] = df['belongs_to_collection'].apply(lambda row:1 if isinstance(row,list) else 0)
# change language to the binary feature 'in_english'

for df in datasets:

    df['in_english'] = df['original_language'].apply(lambda row:1 if row=='en' else 0)
cat_features = ['from_collection', 'day_in_month', 'month', 'day_in_week', 'in_weekend', 'in_english']

for f in cat_features:

    plot_over_time(split=f)

    plot_reveneu_over_feature(f)
numeric_features = ['rating', 'num_spoken_languages', 'num_production_companies', 'num_production_countries',

               'num_Keywords', 'num_cast', 'num_crew', 'lbudget', 'popularity', 'popularity2']

sns.heatmap(train[numeric_features + ['revenue']].corr())
for f in numeric_features:

#     f, ax = plt.subplots(1, 1, figsize=(10,10))

#     sns.regplot(x=f, y='lrevenue',data=train)

    plot_scatter(x=f, y='lrevenue', order=3)
# get all types of genres

def get_all_names(col):

    all_names = []

    for col_list in train[col]:

        for value in col_list:

            if not value['name'] in all_names:

                all_names.append(value['name'])

    return all_names

all_genres = get_all_names('genres')

print('number of genres: ',len(all_genres))
# function to split a dict column to a multihot features

def DicCol2OneHot (df,ColName,all_names=None):

    all_names = get_all_names(ColName) if all_names is None else all_names

#     ColumnDic = {}

#     # Create Combined Dictionary

#     for ColDictionaty in df[ColName]:

#         if bool (ColDictionaty) :

#             #print(ColDictionaty)

#             for Value in ColDictionaty:

#                 ColumnDic[Value[idname]] = Value['name']

    # Create Empty Columns

    for Value in all_names:

        ValueName = ColName + '_' + str(Value)

        df[ValueName] = 0

    for line in df.index:

        for Value in df.loc[line,ColName]:

            if Value['name'] in all_names:

                ValueName = ColName + '_' + str(Value['name'])

                df.loc[line,ValueName] = 1

    return df, all_names
train,all_genres = DicCol2OneHot(train, 'genres')

test,_ = DicCol2OneHot(df, 'genres', all_genres)
# Build genres dataframe that contains important data on each genre:

# this dataframe will contain the following data:

# - number of movies in each genre

# - mean revenue of the movies in this genre

genres = pd.DataFrame(0, index=all_genres, columns=['num_movies', 'mean_revenue'])



for index, row in train.iterrows():

    movie = row['title']

    rev = row['revenue']

    genres_in_movie = [g['name'] for g in row['genres']]

    # save data on genre

    for g in genres_in_movie:

        genres.loc[g]['num_movies'] += 1

        genres.loc[g]['mean_revenue'] += rev



# calculate mean revenue

genres['mean_revenue'] = genres['mean_revenue']/genres['num_movies']

# need to check if its ok to do this operation

# bin actors by profit

q = list(genres.mean_revenue.quantile(q=[0.25, 0.5, 0.75]))

bins = [-np.inf]

bins.extend(q)

bins.append(np.inf)

lables = ['low_profit', 'medium_profit', 'high_profit', 'very_high_profits']

genres['level'] = pd.cut(genres['mean_revenue'], bins=bins, labels=lables, include_lowest=False)
genres.head()
# get all actors names

# actors_names = []

# for cast in train['cast']:

#     for a in cast:

#         if not a['name'] in actors_names:

#             actors_names.append(a['name'])



def get_all_names(col):

    all_names = []

    for col_list in train[col]:

        for value in col_list:

            if not value['name'] in all_names:

                all_names.append(value['name'])

    return all_names

actors_names = get_all_names('cast')

print('number of actors: ',len(actors_names))

# Build actors dataframe that contains important data on each actor:

# this dataframe will contain the following data:

# - number of movies in which the actor participated

# - total med revenue of the movies in which the actor participated

# - number of movies in which the actor participated per genre -- I deleted this column

# - total med revenue of the movies in which the actor participated, per genre -- I deleted this column

actors = pd.DataFrame(0, index=actors_names, columns=['num_movies', 'mean_revenue'])

# # add genre columns to actors 

# for g in all_genres:

#     actors['num_'+g] = 0

#     actors['rev_'+g] = 0



for index, row in train.iterrows():

    movie = row['title']

    cast = row['cast']

    genres = row['genres']

    rev = row['revenue']

    actors_in_movie = [a['name'] for a in cast]

    genres_in_movie = [a['name'] for a in genres]

    # save data on actor

    for a in actors_in_movie:

        actors.loc[a]['num_movies'] += 1

        actors.loc[a]['mean_revenue'] += rev

#         for g in genres_in_movie:

#             actors.loc[a]['num_'+g] += 1

#             actors.loc[a]['rev_'+g] += rev



# calculate mean revenue

actors['mean_revenue'] = actors['mean_revenue']/actors['num_movies']

# for g in all_genres:

#     actors['rev_'+g] = actors['rev_'+g]/actors['num_'+g]

# the profit of an actor is meaningful only of the actor has at least 2 movies:

actors['mean_revenue'] = actors.apply(lambda row: np.nan if row['num_movies'] < 2 else row['mean_revenue'], axis=1)



# bin actors by profit

q = list(actors.mean_revenue.quantile(q=[0.25, 0.5, 0.75]))

bins = [-np.inf]

bins.extend(q)

bins.append(np.inf)

lables = ['low_profit', 'medium_profit', 'high_profit', 'very_high_profits']

actors['level'] = pd.cut(actors['mean_revenue'], bins=bins, labels=lables, include_lowest=False)



actors.head(10)